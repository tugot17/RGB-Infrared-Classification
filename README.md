[![GitHub last commit](https://img.shields.io/github/last-commit/tugot17/RGB-Infrared-Classification)](https://github.com/tugot17/RGB-Infrared-Classification)


# RGB-Infrared image classification
Code for research on the problem of applying transfer-learning for pairs of RGB-Infrared images in image problem

<img src="assets/epfl_classification.png" alt="drawing" width="70%"/>

We investigate which of 5 approaches works best for a selected set of architectures and datasett

## Datasets

The research was conducted on three data sets

* [EPFL RGB NIR Dataset](data/EPFL-thermal-images)

* [SAT4](data/SAT4)

In the `Readme` for each dataset you will find more detailed information how to obtain the exact dataset


## Results

Results for each experiment can be found in respective `Results-summary.ipynb` notebook

## How to run code

First download the Datasets as it is described in [Datasets](#datasets)

and then just run

```
docker-compose up
```
Then copy token from console and run `localhost:8001` with the copied token to access notebooks


If u want to execute something on python scripts level you can access the container by running

```
docker exec -it rgb-infared-classificaton bash
```

where `data-science-project` is the contrainer name

## Authors
* [Piotr Mazurek](https://github.com/tugot17)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
