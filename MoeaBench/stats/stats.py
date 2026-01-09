import MoeaBench as statistic

class stats:

    def __init__(self, result_population):
        self.result_population = result_population


    def indice(self, experiment = None, generation = None):
        ind  = statistic.stats.indice_instance(self.result_population, experiment, generation)
        return ind()
    

    def kstest(self, *args):
        ks = statistic.stats.kstest_instance(args)
        ks()
        return ks
    

    def mwtest(self, *args, alternative):
        mb = statistic.stats.mwtest_instance(args, alternative)
        mb()
        return mb
    

    def paretorank(self,experiment):
        pr = statistic.stats.paretorank_instance(experiment)
        pr()
        return pr