# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# from .I_UserExperiment import I_UserExperiment # Legacy removed
from .base import Reportable
from .run import Run, SmartArray, Population
import numpy as np
import inspect
import os
import numpy as np
import inspect
import os
from ..progress import get_progress_bar, set_active_pbar
from typing import Optional, List, Union, Any, Iterator, Dict
import warnings
import datetime

class JoinedPopulation:
    def __init__(self, pops: List[Population], source: Any = None, label: str = "Population") -> None:
        self.pops = pops
        self.source = source
        self.label = label
        self.name = getattr(source, 'name', 'Experiment') if source else 'Experiment'
        
    @property
    def objectives(self) -> SmartArray:
        if not self.pops:
             return SmartArray(np.array([]), label=self.label, axis_label="Objective")
        data = np.vstack([p.objectives for p in self.pops])
        return SmartArray(data, label=self.label, axis_label="Objective")

    @property
    def variables(self) -> SmartArray:
        if not self.pops:
             return SmartArray(np.array([]), label=self.label, axis_label="Variable")
        data = np.vstack([p.variables for p in self.pops])
        return SmartArray(data, label=self.label, axis_label="Variable")

    @property
    def objs(self) -> SmartArray: return self.objectives

    @property
    def vars(self) -> SmartArray: return self.variables

    def __len__(self) -> int:
        if not self.pops: return 0
        return sum(len(p) for p in self.pops)

class experiment(Reportable):
    def __init__(self, mop: Optional[Any] = None, moea: Optional[Any] = None) -> None:
        self._runs: List[Run] = []
        self._mop: Any = None
        self._moea: Any = None
        self._stop: Any = None
        self._name: str = "experiment"
        self._repeat: int = 1
        
        # Scientific Metadata
        self._authors: Optional[str] = authors if 'authors' in locals() else None
        self._license: str = "GPL-3.0-or-later" 
        self._year: int = datetime.date.today().year

        
        # Use properties for auto-instantiation and validation
        if mop is not None:
            self.mop = mop
        if moea is not None:
            self.moea = moea
        
        # Internal state for execution
        self.result: Any = None 

    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    def __getitem__(self, index: int) -> Run:
        return self._runs[index]

    def __len__(self) -> int:
        return len(self._runs)

    @property
    def name(self) -> str: return self._name
    @name.setter
    def name(self, value: str) -> None: self._name = value
    
    @property
    def year(self) -> int:
        """Returns the publication/experiment year."""
        return self._year
    @year.setter
    def year(self, value: int) -> None:
        """Sets the publication/experiment year."""
        self._year = value

    def _repr_markdown_(self):
        """Rich representation for Jupyter/IPython."""
        return self.report(markdown=True)

    def report(self, **kwargs) -> str:
        """Narrative report of the experiment configuration and metadata."""
        use_md = kwargs.get('markdown', False)
        
        # 1. Name Detection (Smart discovery if default)
        name = self._name
        if name == "experiment":
            try:
                import inspect
                frame = inspect.currentframe().f_back
                for var_name, var_val in frame.f_locals.items():
                    if var_val is self:
                        name = var_name
                        break
            except Exception:
                pass

        # 2. License/Author logic: If no author, force CC0
        license_str = self.license
        if not self.authors or self.authors.strip() == "":
            license_str = "CC0-1.0"

        # 3. Resolve Component Metadata
        mop_name = getattr(self.mop, 'name', self.mop.__class__.__name__) if self.mop else "None"
        moea_name = getattr(self.moea, 'name', self.moea.__class__.__name__) if self.moea else "None"
        
        mop_info = []
        if self.mop:
            if hasattr(self.mop, 'M'): mop_info.append(f"M={self.mop.M}")
            if hasattr(self.mop, 'N'): mop_info.append(f"N={self.mop.N}")
        
        moea_info = []
        if self.moea:
            if hasattr(self.moea, 'population'): moea_info.append(f"Pop={self.moea.population}")
            if hasattr(self.moea, 'generations'): moea_info.append(f"Gens={self.moea.generations}")

        # Meta info
        n_runs = len(self._runs)
        status = "Executed" if n_runs > 0 else "Configured (Not run)"
        
        # Consensus Front Size (Aggregate Cloud Density)
        consensus_line = None
        if n_runs > 1:
            try:
                from ..metrics.evaluator import front_size
                # We calculate for the final generation only to keep report fast
                c_matrix = front_size(self, mode='consensus', gens=-1)
                c_ratio = float(c_matrix.last)
                
                # Total individuals across all runs (final gen)
                total_inds = sum(len(run.pop()) for run in self._runs)
                survived = int(round(c_ratio * total_inds))
                consensus_line = f"{survived}/{total_inds} ({c_ratio*100:.1f}%) globally non-dominated"
            except Exception:
                pass

        if use_md:
            header = f"### Experiment: {name}"
            lines = [
                header,
                f"  - **Status**:    {status}",
                f"  - **Problem**:   {mop_name} ({', '.join(mop_info)})",
                f"  - **Algorithm**: {moea_name} ({', '.join(moea_info)})",
                f"  - **Stop**:      {self.stop or 'Default'}",
                "  - **Metadata**:"
            ]
            lines.extend([
                f"    - Authors: {self.authors or 'Anonymous'}",
                f"    - License: {license_str}",
                f"    - Year:    {self.year}",
                f"    - Runs:    {n_runs} of {self.repeat}"
            ])
            if consensus_line:
                lines.append(f"    - Consensus: {consensus_line}")
                
            return "\n".join(lines)

        # Plain Text
        lines = [
            f"--- Experiment Report: {name} ---",
            f"  Status:    {status}",
            f"  Problem:   {mop_name} ({', '.join(mop_info)})",
            f"  Algorithm: {moea_name} ({', '.join(moea_info)})",
            f"  Stop:      {self.stop or 'Default'}",
            "  Metadata:",
            f"    - Authors: {self.authors or 'Anonymous'}",
            f"    - License: {license_str}",
            f"    - Year:    {self.year}",
            f"    - Runs:    {n_runs} of {self.repeat}"
        ]
        if consensus_line:
            lines.append(f"    - Consensus: {consensus_line}")
            
        return "\n".join(lines)
    @property
    def repeat(self) -> int:
        """Returns the default number of repetitions for this experiment."""
        return self._repeat
    @repeat.setter
    def repeat(self, value: int) -> None:
        """Sets the default number of repetitions for this experiment."""
        if not isinstance(value, int) or value < 1:
            raise ValueError("Repeat must be a positive integer.")
        self._repeat = value
    
    @property
    def mop(self) -> Any: return self._mop
    @mop.setter
    def mop(self, value: Any) -> None:
        # Auto-instantiate if it's a factory (callable but not the final instance with get_CACHE)
        # We assume if it's callable and missing core traits, it's a factory.
        if callable(value) and not hasattr(value, 'get_CACHE'):
             try:
                 self._mop = value()
             except Exception as e:
                 # Fallback or error logging
                 print(f"Warning: Could not instantiate MOP factory: {e}")
                 self._mop = value
        else:
            self._mop = value

    @property
    def moea(self) -> Any: return self._moea
    @moea.setter
    def moea(self, value: Any) -> None:
        self._moea = value

    # Scientific Metadata Properties
    @property
    def authors(self) -> Optional[str]: return self._authors
    @authors.setter
    def authors(self, value: str) -> None: self._authors = value

    @property
    def year(self) -> int: return self._year
    @year.setter
    def year(self, value: int) -> None: self._year = int(value)

    @property
    def license(self) -> str: return self._license
    @license.setter
    def license(self, value: str) -> None:
        # Standard SPDX validation list
        spdx_ids = {
            "GPL-3.0-or-later", "GPL-3.0-only", "GPL-2.0-or-later", "GPL-2.0-only",
            "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "LGPL-3.0-or-later",
            "AGPL-3.0-or-later", "CC0-1.0", "CC-BY-4.0", "MPL-2.0", "Unlicense"
        }
        
        # Simple normalization
        val = value.strip()
        normalization = {
            "GPL3": "GPL-3.0-or-later",
            "GPLv3": "GPL-3.0-or-later",
            "MIT": "MIT",
            "Apache2": "Apache-2.0",
            "BSD": "BSD-3-Clause"
        }
        
        normalized = normalization.get(val.upper(), val)
        if normalized not in spdx_ids:
            warnings.warn(f"'{val}' is not a recognized standard SPDX license identifier. "
                          f"Consider using a standard ID (e.g., 'GPL-3.0-or-later', 'MIT').", 
                          UserWarning)
        
        self._license = normalized

    @property
    def runs(self) -> List[Run]:
        """Returns the list of Run objects from this experiment."""
        return self._runs
        # Original code did magic here to instantiate result.
        # We will handle instantiation in run()

    @property
    def stop(self) -> Any: return self._stop
    @stop.setter
    def stop(self, value: Any) -> None: self._stop = value
    
    @property
    def pof(self) -> Any:
        """Legacy compatibility for save mechanism."""
        return self._mop

    # --- Parameter Delegation to MOEA ---
    @property
    def population(self) -> Optional[int]:
        """Delegates population size access to the MOEA."""
        if hasattr(self.moea, 'population'):
            return self.moea.population
        return None

    @population.setter
    def population(self, value: int) -> None:
        """Delegates population size assignment to the MOEA."""
        if hasattr(self.moea, 'population'):
            self.moea.population = value

    @property
    def generations(self) -> Optional[int]:
        """Delegates generation count access to the MOEA."""
        if hasattr(self.moea, 'generations'):
            return self.moea.generations
        return None

    @generations.setter
    def generations(self, value: int) -> None:
        """Delegates generation count assignment to the MOEA."""
        if hasattr(self.moea, 'generations'):
            self.moea.generations = value

    # Shortcuts
    @property
    def last_run(self) -> Run:
        if not self._runs: raise IndexError("No runs executed yet")
        return self._runs[-1]

    @property
    def last_pop(self) -> Population:
        return self.last_run.last_pop

    def pf(self, n_points: int = 100) -> Any:
        """Returns the true Pareto Front for the current MOP (Shortcut)."""
        return self.mop.pf(n_points)

    def pop(self, gen: int = -1) -> JoinedPopulation:
        """Aggregate populations from all runs."""
        # This is tricky. API.py says:
        # exp.pop(100).objectives # Objs of 100th gen, or all run.
        # This implies returning a "MultiPopulation" or just all of them?
        # If I return a list of Populations, .objectives won't work on the list.
        # The user's API implies exp.pop(100) returns something that has .objectives.
        # Which likely means CONCATENATED objectives from all runs?
        # "metrics.hypervolume(exp)" handles the list iteration.
        # But "exp.pop().objectives" implies aggregation.
        
        # Let's create a JoinedPopulation
        label = self._fmt_label("Population", gen)
        return JoinedPopulation([run.pop(gen) for run in self._runs], source=self, label=label)

    def front(self, gen: int = -1) -> SmartArray:
         """Returns the non-dominated front (objectives) from the aggregate cloud (all runs)."""
         # To get the front of a single run, use: exp.last_run.front()
         return self.superfront(gen)

    def set(self, gen: int = -1) -> SmartArray:
         """Returns the non-dominated decision set (variables) from the aggregate cloud (all runs)."""
         return self.superset(gen)

    def all_fronts(self, gen: int = -1) -> List[SmartArray]:
        """Returns a list of Pareto fronts from all runs."""
        return [run.front(gen) for run in self._runs]

    def all_sets(self, gen: int = -1) -> List[SmartArray]:
        """Returns a list of decision sets from all runs."""
        return [run.set(gen) for run in self._runs]

    def _fmt_label(self, base_label: str, gen: int = -1) -> str:
        """Helper to append generation context to labels if necessary."""
        if gen != -1:
             return f"{base_label}, Gen {gen}"
        return base_label

    def superfront(self, gen: int = -1) -> SmartArray:
        """[Deprecated] Returns the non-dominated front considering all runs combined. Use exp.front() instead."""
        p = self.pop(gen)
        # Create a combined population to apply global filtering
        label = self._fmt_label("Non-dominated", gen)
        combined = Population(p.objectives, p.variables, source=self, label=label)
        res = combined.non_dominated().objectives
        if hasattr(res, 'name'): res.name = self.name or "Experiment"
        return res

    def superset(self, gen: int = -1) -> SmartArray:
        """[Deprecated] Returns the non-dominated decision set considering all runs combined. Use exp.set() instead."""
        p = self.pop(gen)
        label = self._fmt_label("Non-dominated", gen)
        combined = Population(p.objectives, p.variables, source=self, label=label)
        res = combined.non_dominated().variables
        if hasattr(res, 'name'): res.name = self.name or "Experiment"
        return res

    def non_front(self, gen: int = -1) -> SmartArray:
         """Returns the dominated objectives (non-front) from the aggregate cloud."""
         return self.dominated(gen).objectives

    def non_set(self, gen: int = -1) -> SmartArray:
         """Returns the dominated decision set from the aggregate cloud."""
         return self.dominated(gen).variables

    def dominated(self, gen: int = -1) -> Population:
         """Returns the dominated Population from the aggregate cloud at gen."""
         p = self.pop(gen)
         # Filter global cloud for dominated individuals
         label = self._fmt_label("Dominated", gen)
         pop = Population(p.objectives, p.variables, source=self, label=label, gen=gen)
         return pop.dominated()

    def non_dominated(self, gen: int = -1) -> Population:
         """Returns the non-dominated Population from the aggregate cloud at gen."""
         p = self.pop(gen)
         # Filter global cloud for non-dominated individuals
         label = self._fmt_label("Non-dominated", gen)
         pop = Population(p.objectives, p.variables, source=self, label=label, gen=gen)
         return pop.non_dominated()

    def optimal(self, n_points: int = 500) -> Population:
        """Returns a sampling of the true Pareto optimal set and front."""
        if not hasattr(self.mop, 'ps'):
             raise AttributeError(f"MOP {self.mop.__class__.__name__} does not implement ps() sampling.")
        
        # 1. Synchronized Sampling: Get the Pareto Set first
        vars = self.mop.ps(n_points)
        
        # 2. Evaluation: Map the Set to the Front using the problem's own logic
        # This ensures row-by-row correspondence between X and F.
        res = self.mop.evaluation(vars)
        objs = res['F']
        
        # 3. Filtering: Apply non-dominance purely over the theoretical samples
        # This handles cases like DTLZ7 where the condition g=min is not sufficient.
        # For analytical optimal fronts, we want the label to be "Reference" 
        # But we keep the name as the Experiment name so the plotter can do "Exp (Reference)"
        pop = Population(objs, vars, source=self, label="Reference")
        pop = pop.non_dominated()
        
        return pop

    def optimal_front(self, n_points: int = 500) -> SmartArray:
        """Alias for exp.optimal().objectives"""
        return self.optimal(n_points).objectives

    def optimal_set(self, n_points: int = 500) -> SmartArray:
        """Alias for exp.optimal().variables"""
        return self.optimal(n_points).variables

    # Shortcuts from API.py
    @property
    def objectives(self) -> SmartArray:
        return self.pop().objectives

    @property
    def variables(self) -> SmartArray:
        return self.pop().variables

    # Delegation to mop for MOEA compatibility
    @property
    def M(self) -> int: return self.mop.M
    @property
    def N(self) -> int: return self.mop.N
    
    def get_M(self) -> int: return self.mop.get_M()
    def get_Nvar(self) -> Optional[int]: return self.mop.get_Nvar()
    def get_n_ieq_constr(self) -> int: return self.mop.get_n_ieq_constr()

    def evaluation(self, X: np.ndarray, n_ieq_constr: int = 0) -> Dict[str, np.ndarray]:
        """Delegates evaluation to the internal MOP."""
        return self.mop.evaluation(X, n_ieq_constr)
         

    # Execution
    def run(self, repeat: Optional[int] = None, workers: Optional[int] = None, 
            diagnose: bool = False, **kwargs) -> None:
        """
        Executes the optimization experiment for one or more runs.

        Args:
            repeat (int): Number of independent runs to perform. Defaults to self.repeat.
            workers (int): [DEPRECATED] Parallel execution is no longer supported. 
                           All runs are performed serially for stability.
            diagnose (bool): If True, performs automated algorithmic pathology analysis 
                             after execution and prints the rationale. Defaults to False.
            **kwargs: Parameters to override in the MOEA (e.g., generations, population).
        """
        if repeat is not None:
            self.repeat = repeat
        else:
            repeat = self.repeat
        if repeat < 1: repeat = 1
        
        # Propagate overrides to the MOEA
        for key, val in kwargs.items():
            if hasattr(self.moea, key):
                setattr(self.moea, key, val)
        
        # Determine base seed
        base_seed = getattr(self.moea, 'seed', None)
        if base_seed is None:
            base_seed = np.random.randint(0, 1000000)
            if hasattr(self.moea, 'seed'):
                self.moea.seed = base_seed

        # Stop criteria handling
        stop_criteria = kwargs.get('stop', self.stop)
        if hasattr(self.moea, 'stop') and (stop_criteria is not None or 'stop' in kwargs):
            self.moea.stop = stop_criteria

        # Execute serially
        self._run_serial(repeat, base_seed)

        if diagnose and self._runs:
            # Import metrics and diagnostics here to avoid circular dependencies
            from .. import diagnostics, metrics
            
            # Diagnose the last run (most representative of current state)
            run = self.last_run
            try:
                # Calculate quick metrics for diagnosis
                diagnosis_metrics = {}
                
                # Check for Ground Truth
                pf = None
                if hasattr(self.mop, 'pf'):
                    pf = self.mop.pf()
                elif hasattr(self.mop, 'optimal_front'):
                    pf = self.mop.optimal_front()
                elif hasattr(self.mop, 'pareto_front'):
                    pf = self.mop.pareto_front()
                
                if pf is not None:
                     # Calculate GD and IGD for the last population
                     pop_f = run.last_pop.objectives
                     
                     # Using the public metrics API which returns a MetricMatrix
                     # We extract the float value for auditing
                     m_gd = metrics.gd(pop_f, ref=pf)
                     m_igd = metrics.igd(pop_f, ref=pf)
                     
                     diagnosis_metrics['gd'] = float(m_gd)
                     diagnosis_metrics['igd'] = float(m_igd)
                     
                     # Try H_rel if HV available
                     try:
                         m_hv = metrics.hv(pop_f, ref=pf)
                         # Assuming reference set comparison for normalization
                         # For now, we take the raw value as an indicator
                         diagnosis_metrics['h_rel'] = float(m_hv)
                     except:
                         pass
                
                # Perform Audit Using the Textbook Truth Table
                result = diagnostics.audit(diagnosis_metrics)
                result.report_show()
                
            except Exception as e:
                # Silently fail or minimal log to not disrupt main execution
                pass

    def _run_serial(self, repeat: int, base_seed: int) -> None:
        total_gens = getattr(self.moea, 'generations', None)
        outer_pbar = None
        if repeat > 1:
            outer_pbar = get_progress_bar(total=repeat, desc=f"Experiment: {self.name}", position=0)

        for i in range(repeat):
            seed = base_seed + i
            
            inner_pbar = get_progress_bar(total=total_gens, 
                                         desc=f"  Run {i+1}/{repeat}", 
                                         position=1 if repeat > 1 else 0,
                                         leave=False if repeat > 1 else True)
            set_active_pbar(inner_pbar)

            try:
                run_data, _ = self._execute_run(self.moea, self.mop, seed, i+1)
                new_run = Run(run_data, seed, experiment=self, index=i+1)
                self._runs.append(new_run)
            finally:
                inner_pbar.close()
                set_active_pbar(None)
                if outer_pbar:
                    outer_pbar.update_to(i + 1)

        if outer_pbar:
            outer_pbar.close()

    def _execute_run(self, moea, mop, seed, index):
        """Internal helper for executing a single MOEA run."""
        # Inject context
        moea.problem = mop 
        if hasattr(moea, 'seed'):
            moea.seed = seed

        
        try:
            # Execute
            if hasattr(moea, 'evaluation'):
                # New-style API
                data_payload = moea.evaluation()
            else:
                # Legacy Flow
                current_result = moea(mop, None, None, seed)
                raw_result = current_result[0] if isinstance(current_result, tuple) else current_result
                
                if hasattr(raw_result, 'edit_DATA_conf'):
                     moea_instance = raw_result.edit_DATA_conf().get_DATA_MOEA()
                     if hasattr(moea_instance, 'exec'):
                          moea_instance.exec()
                data_payload = raw_result
        except Exception as e:
            # Propagate error if needed
            raise e

        return data_payload, seed

    # Persistence
    def save(self, path: str, mode: str = 'all') -> str:
        from .save import save as legacy_save
        return legacy_save.IPL_save(self, path, mode=mode)

    def load(self, path: str, mode: str = 'all') -> None:
        from .loader import loader
        loader.IPL_loader(self, path, mode=mode)