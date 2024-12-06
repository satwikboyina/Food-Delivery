# from setuptools import find_packages,setup
# from typing import List

# HYPEN_E_DOT='-e .'

# def get_requirements(file_path:str)->List[str]:
#     requirements=[]
#     with open(file_path) as file_obj:
#         requirements=file_obj.readlines()
#         requirements=[req.replace("\n","") for req in requirements]

#         if HYPEN_E_DOT in requirements:
#             requirements.remove(HYPEN_E_DOT)

#     return requirements


# setup(
#     name='FoodDeliveryProject',
#     version='0.0.2',
#     author='Akish',
#     author_email='akishpothuri@gmail.com',
#     install_requires=get_requirements('requirements.txt'),
#     packages=find_packages()
# )

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='FoodDelivery',
    version='0.0.2',
    author='Akish',
    author_email='akishpothuri@gmail.com',
    description='A project for food delivery management',
    license= 'Apache License',
    Platform= 'GIT',
    long_description=open('README.md').read(),  # If you have a README.md
    long_description_content_type='text/markdown',
    url='https://github.com/AKISHPOTHURI/FoodDelivery',  # Replace with your GitHub or project URL
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License',  # Update with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10.2',  # Specify Python version if necessary
)
