from setuptools import setup, find_packages

setup(
    name='charrnobyl',
    version='0.1',
    description='A character-level bigram and neural model for text generation',
    author='Sehaj Ganjoo',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
