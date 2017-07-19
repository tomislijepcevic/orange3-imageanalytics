import logging
from os.path import join, isfile
from itertools import islice

import cachecontrol.caches
import numpy as np
import requests
from Orange.misc.environ import cache_dir

from orangecontrib.textembedding.http2_client import Http2Client
from orangecontrib.textembedding.http2_client import MaxNumberOfRequestsError
from orangecontrib.textembedding.utils import md5_hash
from orangecontrib.textembedding.utils import save_pickle, load_pickle

log = logging.getLogger(__name__)

MODELS = {
    'pubmed': {
        'name': 'PubMed',
        'description': 'CNN model trained on abstracts from MEDLINE.',
        'layers': ['penultimate']
    }
}


class EmbeddingCancelledException(Exception):
    """Thrown when the embedding task is cancelled from another thread.
    (i.e. TextEmbedder.cancelled attribute is set to True).
    """


class TextEmbedder(Http2Client):
    """"Client side functionality for accessing a remote http2
    text embedding backend.

    Examples
    --------
    >>> from orangecontrib.textembedding.text_embedder import TextEmbedder
    >>> texts_file_paths = [...]
    >>> with TextEmbedder(model='model_name', layer='penultimate') as embedder:
    ...    embeddings = embedder(text_file_paths)
    """
    _cache_file_blueprint = '{:s}_{:s}_embeddings.pickle'

    def __init__(self, model, layer, server_url='localhost:8080'):
        super().__init__(server_url)
        model_settings = self._get_model_settings_confidently(model, layer)
        self._model = model
        self._layer = layer

        cache_file_path = self._cache_file_blueprint.format(model, layer)
        self._cache_file_path = join(cache_dir(), cache_file_path)
        self._cache_dict = self._init_cache()

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".TextEmbedder.httpcache"))
        )

        # attribute that offers support for cancelling the embedding
        # if ran in another thread
        self.cancelled = False

    @staticmethod
    def _get_model_settings_confidently(model, layer):
        if model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ', '.join(MODELS.keys())
            raise ValueError(model_error.format(model, available_models))

        model_settings = MODELS[model]

        if layer not in model_settings['layers']:
            layer_error = (
                "'{:s}' is not a valid layer for the '{:s}'"
                " model, should be one of: {:s}")
            available_layers = ', '.join(model_settings['layers'])
            raise ValueError(layer_error.format(
                layer, model, available_layers))

        return model_settings

    def _init_cache(self):
        if isfile(self._cache_file_path):
            try:
                return load_pickle(self._cache_file_path)
            except EOFError:
                return {}

        return {}

    def __call__(self, texts, text_processed_callback=None):
        """Send the texts to the remote server in batches. The batch size
        parameter is set by the http2 remote peer (i.e. the server).

        Parameters
        ----------
        texts: list
            A list of texts to be embedded.

        text_processed_callback: callable (default=None)
            A function that is called after each text is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the text.

        Returns
        -------
        embeddings: array-like
            Array-like of float16 arrays (embeddings) for
            successfully embedded texts and Nones for skipped texts.

        Raises
        ------
        ConnectionError:
            If disconnected or connection with the server is lost
            during the embedding process.

        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        if not self.is_connected_to_server():
            self.reconnect_to_server()

        all_embeddings = []

        for batch in self._yield_in_batches(texts):
            try:
                embeddings = self._send_to_server(
                    batch, text_processed_callback
                )
            except MaxNumberOfRequestsError:
                # maximum number of http2 requests through a single
                # connection is exceeded and a remote peer has closed
                # the connection so establish a new connection and retry
                # with the same batch (should happen rarely as the setting
                # is usually set to >= 1000 requests in http2)
                self.reconnect_to_server()
                embeddings = self._send_to_server(
                    batch, text_processed_callback
                )

            all_embeddings += embeddings
            self.persist_cache()

        return np.array(all_embeddings)

    def _yield_in_batches(self, list_):
        gen_ = (path for path in list_)
        batch_size = self._max_concurrent_streams

        num_yielded = 0

        while True:
            batch = list(islice(gen_, batch_size))
            num_yielded += len(batch)

            yield batch

            if num_yielded == len(list_):
                return

    def _send_to_server(self, texts, text_processed_callback):
        """ Load texts and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        cache_keys = []
        http_streams = []

        for text in texts:
            if self.cancelled:
                raise EmbeddingCancelledException()

            text_bytes = text.encode('utf-8')

            if (len(text_bytes) / 1024).is_integer():
                text += ' '
                text_bytes = text.encode('utf-8')

            cache_key = md5_hash(text_bytes)
            cache_keys.append(cache_key)
            if cache_key in self._cache_dict:
                # skip the sending because text is present in the
                # local cache
                http_streams.append(None)
                continue

            try:
                headers = {
                    'Content-Type': 'text/plain',
                    'Content-Length': str(len(text_bytes))
                }
                stream_id = self._send_request(
                    method='POST',
                    url='/text/' + self._model,
                    headers=headers,
                    body_bytes=text_bytes
                )
                http_streams.append(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise

        # wait for the responses in a blocking manner
        return self._get_responses_from_server(
            http_streams,
            cache_keys,
            text_processed_callback
        )

    def _get_responses_from_server(self, http_streams, cache_keys,
                                   text_processed_callback):
        """Wait for responses from an http2 server in a blocking manner."""
        embeddings = []

        for stream_id, cache_key in zip(http_streams, cache_keys):
            if self.cancelled:
                raise EmbeddingCancelledException()

            if not stream_id:
                # skip rest of the waiting because text was either
                # skipped at loading or is present in the local cache
                embedding = self._get_cached_result_or_none(cache_key)
                embeddings.append(embedding)

                if text_processed_callback:
                    text_processed_callback()
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise

            if not response or 'embedding' not in response:
                # returned response is not a valid json response
                # or the embedding key not present in the json
                embeddings.append(None)
            else:
                # successful response
                embedding = np.array(response['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                self._cache_dict[cache_key] = embedding

            if text_processed_callback:
                text_processed_callback()

        return embeddings

    def _get_cached_result_or_none(self, cache_key):
        if cache_key in self._cache_dict:
            return self._cache_dict[cache_key]
        return None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect_from_server()

    def clear_cache(self):
        self._cache_dict = {}
        self.persist_cache()

    def persist_cache(self):
        save_pickle(self._cache_dict, self._cache_file_path)
