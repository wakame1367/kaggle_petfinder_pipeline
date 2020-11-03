import os


def is_script_running():
    kernel_only_env_keys = {'TERM', 'PAGER', 'GIT_PAGER', 'JPY_PARENT_PID', 'CLICOLOR'}
    kernel_keys = set(os.environ.keys())
    is_kernel = kernel_only_env_keys.issubset(kernel_keys)
    return not is_kernel
