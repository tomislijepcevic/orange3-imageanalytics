import json
import logging
import unittest
from io import BytesIO
from os import environ
from os.path import join, dirname
from unittest.mock import patch

import numpy as np
from h2.exceptions import TooManyStreamsError
from hyper.http20.exceptions import StreamResetError
from numpy.testing import assert_array_equal

from orangecontrib.textembedding.text_embedder import TextEmbedder
from orangecontrib.text.corpus import Corpus

_TESTED_MODULE = 'orangecontrib.textembedding.http2_client.{:s}'
_EXAMPLE_SINGLE = join(dirname(__file__), 'example_pubmed_abstracts.tab')


# dummy classes for hyper.HTTP20Connection monkey patching
class DummyRemoteSettings(object):
    max_concurrent_streams = 128


class DummyH2Connection(object):
    remote_settings = DummyRemoteSettings()


class DummyLockedObject(object):
    _obj = DummyH2Connection()


class DummyHttp2Connection(object):
    def __init__(self, *args, **kwargs):
        self.closed = False
        self._conn = DummyLockedObject()
        self.num_open_streams = 0

    def ping(self, *args, **kwargs):
        if self.closed:
            raise ConnectionRefusedError
        return

    def request(self, *args, **kwargs):
        self.num_open_streams += 1
        max_streams = self._conn._obj.remote_settings.max_concurrent_streams

        if self.num_open_streams > max_streams:
            raise TooManyStreamsError
        if self.closed:
            raise ConnectionRefusedError
        return 1

    def get_response(self, *args, **kwargs):
        if self.closed:
            raise ConnectionResetError
        return BytesIO(json.dumps({'embedding': [0, 1]}).encode())

    def close(self, *args, **kwargs):
        self.closed = True


class TextEmbedderTest(unittest.TestCase):

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.embedder = TextEmbedder(
            model='pubmed',
            layer='penultimate',
            server_url='example.com',
        )
        self.embedder.clear_cache()
        self.single_example = Corpus.from_file(_EXAMPLE_SINGLE)

    def tearDown(self):
        self.embedder.clear_cache()
        logging.disable(logging.NOTSET)

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_connected_to_server(self, ConnectionMock):
        ConnectionMock._discover_server.assert_not_called()
        self.assertEqual(self.embedder.is_connected_to_server(), True)
        # server closes the connection
        self.embedder._server_connection.close()
        self.assertEqual(self.embedder.is_connected_to_server(), False)

    @patch(_TESTED_MODULE.format('HTTP20Connection'))
    def test_connection_errors(self, ConnectionMock):
        res = self.embedder(self.single_example)
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.embedder.clear_cache()

        self.embedder._server_connection.close()
        ConnectionMock.side_effect = ConnectionRefusedError
        with self.assertRaises(ConnectionError):
            self.embedder(self.single_example)

        ConnectionMock.side_effect = BrokenPipeError
        with self.assertRaises(ConnectionError):
            self.embedder(self.single_example)

    @patch.object(DummyHttp2Connection, 'get_response')
    def test_on_stream_reset_by_server(self, ConnectionMock):
        ConnectionMock.side_effect = StreamResetError
        self.assertEqual(self.embedder(self.single_example), [None])
        self.assertEqual(len(self.embedder._cache_dict), 0)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_disconnect_reconnect(self):
        self.assertEqual(self.embedder.is_connected_to_server(), True)
        self.embedder.disconnect_from_server()
        self.assertEqual(self.embedder.is_connected_to_server(), False)
        self.embedder.reconnect_to_server()
        self.assertEqual(self.embedder.is_connected_to_server(), True)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_auto_reconnect(self):
        self.assertEqual(self.embedder.is_connected_to_server(), True)
        self.embedder.disconnect_from_server()
        self.assertEqual(self.embedder.is_connected_to_server(), False)
        self.embedder(self.single_example)
        self.assertEqual(self.embedder.is_connected_to_server(), True)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_on_successful_response(self):
        res = self.embedder(self.single_example)
        assert_array_equal(res, np.array([np.array([0, 1], dtype=np.float16)]))
        self.assertEqual(len(self.embedder._cache_dict), 1)

    @patch.object(
        DummyHttp2Connection, 'get_response',
        lambda self, _: BytesIO(b''))
    def test_on_non_json_response(self):
        self.assertEqual(self.embedder(self.single_example), [None])
        self.assertEqual(len(self.embedder._cache_dict), 0)

    @patch.object(
        DummyHttp2Connection, 'get_response',
        lambda self, _: BytesIO(json.dumps({'wrong_key': None}).encode()))
    def test_on_json_wrong_key_response(self):
        self.assertEqual(self.embedder(self.single_example), [None])
        self.assertEqual(len(self.embedder._cache_dict), 0)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_persistent_caching(self):
        self.assertEqual(len(self.embedder._cache_dict), 0)
        self.embedder(self.single_example)
        self.assertEqual(len(self.embedder._cache_dict), 1)

        self.embedder.persist_cache()
        self.embedder = TextEmbedder(
            model='pubmed',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(self.embedder._cache_dict), 1)

        self.embedder.clear_cache()
        self.embedder = TextEmbedder(
            model='pubmed',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(self.embedder._cache_dict), 0)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_with_statement(self):
        with self.embedder as embedder:
            embedder(self.single_example)

        self.assertEqual(self.embedder.is_connected_to_server(), False)
        self.embedder = TextEmbedder(
            model='pubmed',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertEqual(len(self.embedder._cache_dict), 1)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_max_concurrent_streams_setting(self):
        self.assertEqual(self.embedder._max_concurrent_streams, 128)

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            self.embedder = TextEmbedder(
                model='invalid_model',
                layer='penultimate',
                server_url='example.com',
            )

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_invalid_layer(self):
        with self.assertRaises(ValueError):
            self.embedder = TextEmbedder(
                model='pubmed',
                layer='first',
                server_url='example.com',
            )

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_server_url_env_var(self):
        url_value = 'url:1234'
        self.assertTrue(self.embedder._server_url != url_value)

        environ['ORANGE_EMBEDDING_API_URL'] = url_value
        self.embedder = TextEmbedder(
            model='pubmed',
            layer='penultimate',
            server_url='example.com',
        )
        self.assertTrue(self.embedder._server_url == url_value)
        del environ['ORANGE_EMBEDDING_API_URL']

    @patch(_TESTED_MODULE.format('HTTP20Connection'), DummyHttp2Connection)
    def test_embedding_cancelled(self):
        self.assertFalse(self.embedder.cancelled)
        self.embedder.cancelled = True
        with self.assertRaises(Exception):
            self.embedder(self.single_example)
