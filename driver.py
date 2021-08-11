import os
##add arg parser

from src.experiments.removal_experiments import RemovalExperiments
from src.segmentation.segmentation import Segmentation

def main():

	removal_experiments = RemovalExperiments()

	removal_experiments.run_removal_experiments()

if __name__ == '__main__':
	main()
