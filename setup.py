from setuptools import setup, find_packages

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]
setup(
    name='Start_up',
    version='0.0.1',
    author='Deepika Venugopal',
    author_email='deepikavenugopal17@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ]
)