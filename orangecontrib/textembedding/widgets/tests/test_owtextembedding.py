import os

from Orange.data import Table
from orangecontrib.text.corpus import Corpus, get_sample_corpora_dir
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.textembedding.widgets.owtextembedding import OWTextEmbedding


class TestOWTextEmbedding(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.signal_name = "Corpus"
        cls.signal_data = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWTextEmbedding)

    def test_not_text_data(self):
        """
        It should not fail when there is a data without images.
        GH-45
        GH-46
        """
        table = Table("iris")
        self.send_signal("Corpus", table)

    def test_none_data(self):
        """
        It should not fail when there is no data.
        GH-46
        """
        self.send_signal("Corpus", None)
