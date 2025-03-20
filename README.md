# Knowledge-Guided Hybrid Deep Reinforcement Learning for Dynamic Multi-Depot Electric Vehicle Routing Problem

This repository contains code and resources for the paper:

**"Knowledge-Guided Hybrid Deep Reinforcement Learning for Dynamic Multi-Depot Electric Vehicle Routing Problem"**  
Authors: Reza Shahbazian, Francesca Guerriero, Alessia Ciacco, Giusy Macrina  
Affiliation: Department of Mechanical, Energy, and Management Engineering, University of Calabria

---

## Table of Contents

- [Introduction](#introduction)
- [Problem Description](#problem-description)
- [Proposed Solution](#proposed-solution)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Introduction

The project focuses on addressing the *Dynamic Multi-Depot Electric Vehicle Routing Problem with Time Windows* (DMDEVRPTW), where electric vehicles (EVs) must adapt routes dynamically across multiple depots, serving customers within designated time windows while considering battery and energy constraints. The framework integrates a hybrid approach combining deep reinforcement learning (DRL) and variable neighborhood search (VNS), guided by domain knowledge, to enhance solution quality, efficiency, and scalability.

## Problem Description

DMDEVRPTW is a challenging optimization problem due to the complexities introduced by:
- **Dynamic Environment**: Customer demands, travel times, and service times vary in real time.
- **Multiple Depots**: Vehicles operate from different depots, with depot-specific constraints.
- **Time Windows**: Customers must be served within specified time periods.
- **Battery Constraints**: EVs have limited battery capacities and may need recharging at specific points.

The objective is to minimize the total traveled distance and time windows violations, while optimizing route selection to handle dynamic changes effectively.

## Proposed Solution

The framework utilizes a hybrid approach:
1. **Knowledge-Guided Deep Reinforcement Learning (DRL)**: A Double Deep Q-Network (DDQN) is used for initial route generation. Knowledge, represented as *State-Action Permissibility*, is incorporated to reduce overestimation and guide DRL agents towards high-quality solutions.
2. **Variable Neighborhood Search (VNS)**: After DRL-based route generation, VNS refines the routes through local search, focusing on minimizing travel distance and time penalty costs.

### Key Features
- **Prioritized Experience Replay**: Enhances the learning efficiency of DDQN.
- **Regional Knowledge-Based Clustering**: Partitions the environment based on customer-depot proximity, optimizing route assignments.
- **Multi-Agent Cooperation**: Shared knowledge among EVs enhances efficiency in multi-depot scenarios.
  
## Installation

To use this codebase, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/ShahbazianR/VRP.git
cd VRP
pip install -r requirements.txt


## Usage

1. **Configuration**: Configure parameters such as the number of depots, vehicles, customer demands, and time windows in the configuration file.
2. **Training**: Train the DRL model to generate initial routes.
3. **Execution**: Run the DRL model, followed by VNS to optimize the route.


## Results

Extensive testing on real-world and benchmark datasets shows:
- **Over 70% reduction in route distance** compared to state-of-the-art methods.
- Improved computational efficiency and scalability across various problem instances.

For more detailed results, please refer to the paper.

## References

If you use this repository, please cite the following paper:

Shahbazian, R., Guerriero, F., Ciacco, A., & Macrina, G. "Knowledge-Guided Hybrid Deep Reinforcement Learning for Dynamic Multi-Depot Electric Vehicle Routing Problem". *Submitted to Computer and Operations Research*, 2024.
