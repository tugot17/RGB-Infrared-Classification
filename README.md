[![GitHub last commit](https://img.shields.io/github/last-commit/tugot17/Data-Science-Project-Template)](https://github.com/tugot17/Data-Science-Project-Template/)

# Data-Science-Project-Template
Universal template for your datascience project. 

As a backbone we use [scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) image, which probably aready contains the data-science package that you are interested in. 

## How to run it

Just run

```
docker-compose up
```
Then copy token from console and run `localhost:8001` with the copied token to access notebooks


If u want to execute something on python scripts level you can access the container by running

```
docker exec -it data-science-project bash
```

where `data-science-project` is the contrainer name

## Authors
* [Piotr Mazurek](https://github.com/tugot17)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
