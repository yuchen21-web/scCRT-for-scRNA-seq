from distutils.core import setup

setup(
    # package_dir={'':'scCRT'},
    packages=['model'],
    name="scCRT",
    version="1.0",
    description="a dimensionality reduction model for scRNA-seq trajectory inference",
    author="Yuchen Shi",
    author_email="yuchen@hdu.edu.cn",
    py_modules=["scCRTUtils"]
    )
