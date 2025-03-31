from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'
def getrequirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements

    '''
    reuirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
name ='MLProject',
version='0.0.1',
author='Swati',
author_email='singhkendall14@gmail.com',
packages=find_packages(),
## In certain cases we would reuire a lot of libraries to install, hence it is better to use functions
##install_requires =['pandas','numpy','seaborn']
install_requires = getrequirements('requirements.txt')

)