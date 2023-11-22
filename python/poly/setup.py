from setuptools import setup, find_packages

setup(
    name='poly',
    version='0.0.1',
    description='Polynomial API',
    author=['yongwoo', 'seonyoung'],
    author_email=['dragonrain96@gmail.com', 'seonyoung@yonsei.ac.kr'],
    install_requires=['numpy', 'hecate'],
    packages=find_packages(exclude=[]),
    include_package_data=True,
    keywords=['yongwoo', 'seonyoung','homomorphic encryption', 'ckks', 'hecate', 'elasm', 'dacapo'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
