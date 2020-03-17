import os
import platform
import zipfile
import shutil
import getpass

def main():
    print('Setting up SLAK')

    if 'ubuntu' in platform.platform().lower():
        apt_get_cmd = 'sudo apt-get install cmake build-essential nvidia-cuda-toolkit git-lfs'
        print('Going to run command:\n \"' + apt_get_cmd + '\"')
        print('which will require your password')
        os.system(apt_get_cmd)

    os.system('git submodule update --init --recursive')
    
    print("\nExtracting...")
    with zipfile.ZipFile('./grsidata/SLAKData.zip') as meshes:
        meshes.extractall('./grsidata/', )
    print('[DONE]\n')

    if os.path.exists('./build'):
        shutil.rmtree('./build')

    os.mkdir('./build')
    os.chdir('./build')

    cmake_command = 'cmake -DCUDA_BUILD_CC61:BOOL=ON -DCUDA_BUILD_CC70_SYNC:BOOL=ON -DCUDA_BUILD_CC75_SYNC:BOOL=ON ..'
    print('Running CMake command:\n \"' + cmake_command + '\"')
    os.system(cmake_command)

    current_os = platform.system()

    if current_os != 'Windows' and current_os != 'Linux':
        print('\nAuto build and run not supported for current OS.')
        print('Please build the program manually and run with \"SLAK ./data/config.txt\"')
        raise SystemExit

    os.system('cmake --build ./ --config Release')

    exec_extension = ''
    exec_prefix = ''
    if current_os == 'Windows':
        shutil.copyfile('./Release/SLAK.exe', 'SLAK.exe')
        exec_prefix = '.\\'
        exec_extension = '.exe'
    
    if current_os == 'Linux':
        os.system('chmod +x SLAK')
        exec_prefix = './'
        
    print(exec_prefix + 'SLAK' + exec_extension + ' ../grsidata/config.txt')
    os.system(exec_prefix + 'SLAK' + exec_extension + ' ../grsidata/config.txt')


if __name__ == "__main__":
    main()