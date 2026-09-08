from setuptools import find_packages, setup


setup(
    name="crane_llm",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "jupyter-builder>=1.0.0,<2",
    ],
    package_data={
        "nb_extension": [
            "static/*.js",
            "package.json",
            "tsconfig.json",
            "src/*.ts",
        ],
    },
    classifiers=[
        "Framework :: Jupyter",
        "Framework :: Jupyter :: JupyterLab",
        "Framework :: Jupyter :: JupyterLab :: 4",
        "Framework :: Jupyter :: JupyterLab :: Extensions",
        "Framework :: Jupyter :: JupyterLab :: Extensions :: Prebuilt",
    ],
)
