from analyzer import OMVAnalyzer

class MorphologyAnalyzer(OMVAnalyzer):
    
    def before_running(self):
        if 'total area' in self.expected:
            base = self.observable.get('base section', 'cell[0]')
            self.query = self.backend.query_area(base)

    def parse_expected(self):
        return self.expected['total area']

    def parse_observable(self):
        area = float(self.backend.fetch_query(self.query))
        scale = float(self.observable.get('scaling', 1))
        return scale * area


















