conda create --name chrislab python=3.10 -y; conda activate chrislab;
rm -rf build dist src/*.egg-info;
pip install build; python3 -m build;
pip install twine; python3 -m twine upload dist/*;
rm -rf build dist src/*.egg-info;

conda create --name chrislab python=3.10 -y; conda activate chrislab;
pip install --upgrade chrislab; pip list;
