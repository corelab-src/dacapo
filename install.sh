cd $HECATE/python/hecate
python setup.py sdist --format=tar
pip uninstall hecate
pip install dist/hecate-0.0.1.tar

cd $HECATE/python/poly
python setup.py sdist --format=tar
pip uninstall poly
pip install dist/poly-0.0.1.tar
