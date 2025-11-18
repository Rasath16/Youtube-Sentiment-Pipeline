from setuptools import setup, find_packages

setup(
    name="youtube-sentiment-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "mlflow",
        "nltk",
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
    ],
)