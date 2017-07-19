import os
import logging
import traceback
from types import SimpleNamespace as namespace

import numpy as np
from AnyQt.QtCore import Qt, QTimer, QThread, QThreadPool
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtTest import QSignalSpy
from AnyQt.QtWidgets import QLayout, QPushButton, QStyle

from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.gui import hBox
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils import concurrent as qconcurrent
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Default

from orangecontrib.textembedding.text_embedder import TextEmbedder
from orangecontrib.textembedding.text_embedder import MODELS as EMBEDDERS_INFO

class _Input:
    DATA = 'Data'


class _Output:
    EMBEDDINGS = 'Embeddings'


class OWTextEmbedding(OWWidget):
    name = "Text Embedding"
    description = "Text embedding through deep neural networks."
    icon = "icons/TextEmbedding.svg"
    priority = 150

    want_main_area = False
    _auto_apply = Setting(default=True)

    inputs = [(_Input.DATA, Table, 'set_data')]
    outputs = [(_Output.EMBEDDINGS, Table, Default)]

    cb_text_attr_current_id = Setting(default=0)
    cb_embedder_current_id = Setting(default=0)

    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        super().__init__()
        self.embedders = sorted(list(EMBEDDERS_INFO))
        self._text_attributes = None
        self._input_data = None
        self._log = logging.getLogger(__name__)
        self._task = None
        self._setup_layout()
        self._text_embedder = None
        self._executor = qconcurrent.ThreadExecutor(
            self, threadPool=QThreadPool(maxThreadCount=1)
        )
        self.setBlocking(True)
        QTimer.singleShot(0, self._init_server_connection)

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, 'Info')
        self.input_data_info = widgetLabel(widget_box, self._NO_DATA_INFO_TEXT)
        self.connection_info = widgetLabel(widget_box, "")

        widget_box = widgetBox(self.controlArea, 'Settings')
        self.cb_text_attr = comboBox(
            widget=widget_box,
            master=self,
            value='cb_text_attr_current_id',
            label='Text attribute:',
            orientation=Qt.Horizontal,
            callback=self._cb_text_attr_changed
        )

        self.cb_embedder = comboBox(
            widget=widget_box,
            master=self,
            value='cb_embedder_current_id',
            label='Embedder:',
            orientation=Qt.Horizontal,
            callback=self._cb_embedder_changed
        )
        self.cb_embedder.setModel(VariableListModel(
            [EMBEDDERS_INFO[e]['name'] for e in self.embedders]))
        if not self.cb_embedder_current_id < len(self.embedders):
            self.cb_embedder_current_id = 0
        self.cb_embedder.setCurrentIndex(self.cb_embedder_current_id)

        current_embedder = self.embedders[self.cb_embedder_current_id]
        self.embedder_info = widgetLabel(
            widget_box,
            EMBEDDERS_INFO[current_embedder]['description']
        )

        self.auto_commit_widget = auto_commit(
            widget=self.controlArea,
            master=self,
            value='_auto_apply',
            label='Apply',
            commit=self.commit
        )

        self.cancel_button = QPushButton(
            'Cancel',
            icon=self.style().standardIcon(QStyle.SP_DialogCancelButton),
        )
        self.cancel_button.clicked.connect(self.cancel)
        hbox = hBox(self.controlArea)
        hbox.layout().addWidget(self.cancel_button)
        self.cancel_button.hide()

    def _init_server_connection(self):
        self.setBlocking(False)
        self._text_embedder = TextEmbedder(
            model=self.embedders[self.cb_embedder_current_id],
            layer='penultimate'
        )
        self._set_server_info(
            self._text_embedder.is_connected_to_server()
        )

    def set_data(self, data):
        if not data:
            self._input_data = None
            self.send(_Output.EMBEDDINGS, None)
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)
            return

        self._text_attributes = self._filter_text_attributes(data)
        if not self._text_attributes:
            input_data_info_text = (
                "Data with {:d} instances, but without text attributes."
                .format(len(data)))
            input_data_info_text.format(input_data_info_text)
            self.input_data_info.setText(input_data_info_text)
            self._input_data = None
            return

        if not self.cb_text_attr_current_id < len(self._text_attributes):
            self.cb_text_attr_current_id = 0

        self.cb_text_attr.setModel(VariableListModel(self._text_attributes))
        self.cb_text_attr.setCurrentIndex(self.cb_text_attr_current_id)

        self._input_data = data
        input_data_info_text = "Data with {:d} instances.".format(len(data))
        self.input_data_info.setText(input_data_info_text)

        self._cb_text_attr_changed()

    @staticmethod
    def _filter_text_attributes(data):
        metas = data.domain.metas
        return [ m for m in metas if m.is_string ]

    def _cb_text_attr_changed(self):
        self.commit()

    def _cb_embedder_changed(self):
        current_embedder = self.embedders[self.cb_embedder_current_id]
        self._text_embedder = TextEmbedder(
            model=current_embedder,
            layer='penultimate'
        )
        self.embedder_info.setText(
            EMBEDDERS_INFO[current_embedder]['description'])
        self.commit()


    def commit(self):
        if self._task is not None:
            self.cancel()

        if self._text_embedder is None:
            self._set_server_info(connected=False)
            return

        if not self._text_attributes or self._input_data is None:
            self.send(_Output.EMBEDDINGS, None)
            return

        self._set_server_info(connected=True)
        self.cancel_button.show()
        self.cb_text_attr.setDisabled(True)
        self.cb_embedder.setDisabled(True)

        texts_attr = self._text_attributes[self.cb_text_attr_current_id]
        texts = self._input_data[:, texts_attr].metas.flatten()

        ticks = iter(np.linspace(0.0, 100.0, len(texts)))
        set_progress = qconcurrent.methodinvoke(
            self, "__progress_set", (float,))

        def advance():
            set_progress(next(ticks))

        def cancel():
            task.future.cancel()
            task.cancelled = True
            task.embedder.cancelled = True

        embedder = self._text_embedder

        def run_embedding(texts):
            return embedder(texts=texts, text_processed_callback=advance)

        self.auto_commit_widget.setDisabled(True)
        self.progressBarInit(processEvents=None)
        self.progressBarSet(0.0, processEvents=None)
        self.setBlocking(True)

        f = self._executor.submit(run_embedding, texts)
        f.add_done_callback(
            qconcurrent.methodinvoke(self, "__set_results", (object,)))

        task = self._task = namespace(
            texts=texts,
            embedder=embedder,
            cancelled=False,
            cancel=cancel,
            future=f,
        )
        self._log.debug("Starting embedding task for %i texts", len(texts))
        return

    @Slot(float)
    def __progress_set(self, value):
        assert self.thread() is QThread.currentThread()
        if self._task is not None:
            self.progressBarSet(value)

    @Slot(object)
    def __set_results(self, f):
        assert self.thread() is QThread.currentThread()
        if self._task is None or self._task.future is not f:
            self._log.info("Reaping stale task")
            return

        assert f.done()

        task, self._task = self._task, None
        self.auto_commit_widget.setDisabled(False)
        self.cancel_button.hide()
        self.cb_text_attr.setDisabled(False)
        self.cb_embedder.setDisabled(False)
        self.progressBarFinished(processEvents=None)
        self.setBlocking(False)

        try:
            embeddings = f.result()
        except ConnectionError:
            self._log.exception("Error", exc_info=True)
            self.send(_Output.EMBEDDINGS, None)
            self._set_server_info(connected=False)
            return
        except Exception as err:
            self._log.exception("Error", exc_info=True)
            self.error("\n".join(traceback.format_exception_only(type(err), err)))
            self.send(_Output.EMBEDDINGS, None)
            return

        assert self._input_data is not None
        assert len(self._input_data) == embeddings.shape[0]

        self._send_output_signals(embeddings)

    def _send_output_signals(self, embeddings):
            embedded_texts = self._construct_output_data_table(
                self._input_data,
                embeddings
            )
            embedded_texts.ids = self._input_data.ids

            self.send(_Output.EMBEDDINGS, embedded_texts)

    @staticmethod
    def _construct_output_data_table(embedded_texts, embeddings):
        X = embeddings
        Y = embedded_texts.Y

        attributes = [ContinuousVariable.make('n{:d}'.format(d))
                      for d in range(embeddings.shape[1])]
        attributes = list(embedded_texts.domain.attributes) + attributes

        domain = Domain(
            attributes=attributes,
            class_vars=embedded_texts.domain.class_vars,
            metas=embedded_texts.domain.metas
        )

        return Table(domain, X, Y, embedded_texts.metas)

    def _set_server_info(self, connected):
        self.clear_messages()
        if connected:
            self.connection_info.setText("Connected to server.")
        else:
            self.connection_info.setText("No connection with server.")
            self.warning("Click Apply to try again.")

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()
        self._text_embedder.__exit__(None, None, None)

    def cancel(self):
        if self._task is not None:
            task, self._task = self._task, None
            task.cancel()
            # wait until done
            try:
                task.future.exception()
            except qconcurrent.CancelledError:
                pass

            self.auto_commit_widget.setDisabled(False)
            self.cancel_button.hide()
            self.progressBarFinished(processEvents=None)
            self.setBlocking(False)
            self.cb_text_attr.setDisabled(False)
            self.cb_embedder.setDisabled(False)
            self._text_embedder.cancelled = False
            # reset the connection.
            connected = self._text_embedder.reconnect_to_server()
            self._set_server_info(connected=connected)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()

    if len(argv) > 1:
        file_path = argv[1]
    else:
        file_path = 'tests/pubmed_dataset.tab'

    data = Table(file_path)
    widget = OWTextEmbedding()
    widget.show()
    assert QSignalSpy(widget.blockingStateChanged).wait()
    widget.set_data(data)
    widget.handleNewSignals()
    app.exec()
    widget.set_data(None)
    widget.handleNewSignals()
    widget.saveSettings()
    widget.onDeleteWidget()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
