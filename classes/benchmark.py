from classes.benchmark_static import BenchmarkStatic
from classes.benchmark_temporal import BenchmarkTemporal

class Benchmark(BenchmarkStatic,BenchmarkTemporal):


    def __init__(self, G, Gs):
        super().__init__(G, Gs)

    