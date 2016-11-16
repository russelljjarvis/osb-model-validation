import os
import subprocess as sp

from ..common.inout import inform, trim_path, check_output, is_verbose
from engine import OMVEngine, EngineExecutionError

from nestsli import NestEngine


class PyNestEngine(OMVEngine):
    
    name = "PyNEST"
    
    @staticmethod
    def is_installed(version):
        
        PyNestEngine.environment_vars = NestEngine.get_nest_environment()
        
        ret = True
        try:
            
            ret_str = check_output(['python -c "import nest; print(nest.version())"'], shell=True)
            ret = len(ret_str) > 0
            
            if ret and is_verbose():
                inform("%s is correctly installed..." % (PyNestEngine.name), indent=2)
            
            if is_verbose():
                inform("NEST is installed with Python support...", indent=2)
                inform("Env vars: %s" % PyNestEngine.environment_vars, indent=2)
            
        except Exception as err:
            inform("Couldn't import NEST into Python: ", err, indent=1)
            inform("NEST env vars: %s" % PyNestEngine.environment_vars, indent=1)
            ret = False
        return ret
        
        
    @staticmethod
    def install(version):
        NestEngine.install(version)
        PyNestEngine.path = NestEngine.path
        PyNestEngine.environment_vars = NestEngine.environment_vars
        
        
    def run(self):
        
        self.environment_vars = NestEngine.get_nest_environment()
        self.set_environment()
                                        
        inform("Env vars: %s" % self.environment_vars, indent=2)
        
        try:
            inform("Running file %s with %s" % (trim_path(self.modelpath), self.name), indent=1)
            self.stdout = check_output(['python', self.modelpath],
                                          cwd=os.path.dirname(self.modelpath))
            self.returncode = 0
        except sp.CalledProcessError as err:
            self.returncode = err.returncode
            self.stdout = err.output
            raise EngineExecutionError
        except Exception as err:
            inform("Another error with running %s: "%self.name, err, indent=1)
            self.returncode = -1
            self.stdout = "???"


















