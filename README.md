# air-polution-sensor
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align = "center">
  <a href = "https://github.com/Johansmm/air-polution-sensor">
    <img src="https://github.com/Johansmm/air-polution-sensor/blob/main/results/signal_travel.gif" alt="Logo" width="720" height="480">
  </a>

  <h3 align="center">Analyse spatio-temporelle de signaux pour détecter la dérivede capteurs mesurant la qualité de l’air</h3>

  <p align="center">
    Detection of drift concept in a sensor network from the Graph Fourier Transform (GFT).
    <br />
    <a href="https://github.com/Johansmm/air-polution-sensor"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Johansmm/air-polution-sensor/tree/main/results">View Demo</a>
    ·
    <a href="https://github.com/Johansmm/air-polution-sensor/issues">Report Bug</a>
    ·
    <a href="https://github.com/Johansmm/air-polution-sensor/issues">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
The network of sensors allows to follow a complex phenomenon by observing the temporal evolution of the values in several points, and by crossing the information from one sensor to another. This allows, for example, to carry out a meteorological or seismic monitoring, by detecting a cloud or a tremor. However, the sensors used are sometimes subject to drift, wear and tear or possible interference, which makes some of the observed values false. It is therefore important to be able to determine if a sensor starts to drift in order to allow a good analysis of the data. Therefore, the objective of this project is to detect a drifting sensor (when and where) in a real sensor's network. For this, we will explore the usefulness of a particular mathematical tool: the graph space-time spectrogram. Such a tool allows to decompose a space-time series into a sum of space-time frequencies, in a similar way to Fourier analysis, but for signals in graphs, known as Graph Fourier Transform (GFT).

### Built With
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](http://pandas.pydata.org/)
* [pygsp](https://pygsp.readthedocs.io/en/stable/)
* [pygraphviz](https://pygraphviz.github.io/)
* [fancyimpute](https://pypi.org/project/fancyimpute/)

<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running follow these simple steps.

### Prerequisites
It's possible that need install the following code
```
sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/daca1897/Age-Detection-Actors-Challenge.git
   ```
2. Install requerements
   ```sh
   python3 -m pip install -U requerements.
   ```

<!-- USAGE EXAMPLES -->
## Usage
Run the different notebooks that is contained in this repository. The main objectives of the project were developed in the **sensors_drift.ipynb** notebook. However, you can use the following notebooks:

* [`examples/First_spectogram_tests.ipynb`](https://github.com/Johansmm/air-polution-sensor/tree/main/examples/First_spectogram_tests.ipynb) : Manipulation of pygsp library. Acknowledgments to [BastienPasdeloup](https://github.com/BastienPasdeloup).
* [`examples/GFT_sensors.ipynb`](https://github.com/Johansmm/air-polution-sensor/tree/main/examples/GFT_sensors.ipynb) : Presentation of a network and synthetic signals in order to visualize some concepts such as GFT, spectrograms and drift concept.

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/Johansmm/air-polution-sensor/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
{:refdef: style="width: 10px; height: 10px"}
* Johan Mejia (johan-steven.mejia-mogollon@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-1]
* Tatiana Moreno (jenny-tatiana.moreno-perea@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-2]
* Diego Carreño (diego-andres.carreno-avila@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-3]
* Ilias Amal (ilias.amal@imt-atlantique.net) - [![Linkend][linkedin-shield]][linkedin-url-4]
* Project Link: [https://github.com/Johansmm/air-polution-sensor](https://github.com/Johansmm/air-polution-sensor)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
The authors would like to thank [AlexandreReiffers](https://alreiff.github.io/index.html) and [BastienPasdeloup](https://github.com/BastienPasdeloup) for their support and follow-up during the development of the project.

We also thank [Best-README-Template](https://github.com/othneildrew/Best-README-Template) for providing the repository documentation. 

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Johansmm/air-polution-sensor.svg?style=for-the-badge
[contributors-url]: https://github.com/Johansmm/air-polution-sensor/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Johansmm/air-polution-sensor.svg?style=for-the-badge
[forks-url]: https://github.com/Johansmm/air-polution-sensor/network/members
[stars-shield]: https://img.shields.io/github/stars/Johansmm/air-polution-sensor.svg?style=for-the-badge
[stars-url]: https://github.com/Johansmm/air-polution-sensor/stargazers
[issues-shield]: https://img.shields.io/github/issues/Johansmm/air-polution-sensor.svg?style=for-the-badge
[issues-url]: https://github.com/Johansmm/air-polution-sensor/issues
[license-shield]: https://img.shields.io/github/license/Johansmm/air-polution-sensor.svg?style=for-the-badge
[license-url]: https://github.com/Johansmm/air-polution-sensor/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-blue.svg?style=plastice&logo=linkedin

[linkedin-url-1]: https://www.linkedin.com/in/johansmm/
[linkedin-url-2]: https://www.linkedin.com/in/tatiana-moreno-perea/
[linkedin-url-3]: https://www.linkedin.com/in/diego-andres-carre%C3%B1o-49b2ab157/
[linkedin-url-4]: https://www.linkedin.com/in/ilias-amal-455502183/

