import os
from omv.common.inout import inform, check_output
from omv.engines.utils.wdir import working_dir

def install_numba():
    try:
        print(check_output(['pip', 'install', 'numba']))  # This should ideally be automatically installed with PyNN...
        print(check_output(['pip', 'install', 'quantities']))  # This should ideally be automatically installed with PyNN...
        print(check_output(['pip', 'install', 'neo==0.5.1']))  # This should ideally be automatically installed with PyNN...

        install_root = os.environ['HOME']

        numba_src = 'numba_src'
        with working_dir(install_root):
            check_output(['git', 'clone', 'https://github.com/russelljjarvis/jit_hub', numba_src])

        path = os.path.join(install_root, numba_src)

        with working_dir(path):
            print(check_output(['git','checkout','main']))  # neuroml branch has the latest NML2 import/export code!
            print(check_output(['python', 'setup.py', 'install']))
            print(check_output(['pwd']))
            print("Finished attempting to install numba jithub")
        import jithub
        m = 'Successfully installed numba jithub...'
    except Exception as e:
        m = 'ERROR during install jithub: %s'%e
    finally:
        inform(m)
