from os.path import join, dirname

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.textembedding.widgets.owtextembedding import OWTextEmbedding


class TestOWTextEmbedding(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.signal_name = "Data"
        cls.signal_data = Table(join(dirname(__file__), 'pubmed_dataset.tab'))

    def setUp(self):
        self.widget = self.create_widget(OWTextEmbedding)

    def test_not_text_data(self):
        """
        It should not fail when there is a data without images.
        GH-45
        GH-46
        """
        self.send_signal("Data", self.signal_data)

    def test_none_data(self):
        """
        It should not fail when there is no data.
        GH-46
        """
        self.send_signal("Data", None)
