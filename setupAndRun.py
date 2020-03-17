import os
import platform
import urllib2
import zipfile
import shutil
import getpass

def main():
    print('Setting up SLAK')

    print("Downloading Meshes...")
    url = 'https://files.icg.tugraz.at/f/997236cfb9a44d02af66/?dl=1'
    fdata = urllib2.urlopen(url)
    with open('./data/SLAKData.zip', 'wb') as f:
        f.write(fdata.read())
    print("[DONE]\n")

    password = getpass.getpass(prompt='Please enter the archive password: ', stream=None) 
    print("\nExtracting...")
    with zipfile.ZipFile('./data/SLAKData.zip') as meshes:
        meshes.extractall('./data/', pwd=password.encode())
    print('[DONE]\n')
    os.remove('./data/SLAKData.zip')

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
        
    print(exec_prefix + 'SLAK' + exec_extension + ' ../data/config.txt')
    os.system(exec_prefix + 'SLAK' + exec_extension + ' ../data/config.txt')


if __name__ == "__main__":
    main()