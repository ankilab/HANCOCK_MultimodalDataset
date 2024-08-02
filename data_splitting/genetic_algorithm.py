from argparse import ArgumentParser
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import imageio.v2 as io
from natsort import natsorted
from pathlib import Path
import os
from data_exploration.umap_embedding import setup_preprocessing_pipeline, get_umap_embedding


SEED = 42


class Individual:
    def __init__(self, chromosome=None, random_num_gen=None, mutation_operator="inversion", target="in"):
        """
        An Individual represents a single split into training and test data.

        Parameters
        ----------
        chromosome : numpy.ndarray
            Array with length equal to #points where 0 indicates training and 1 indicates test
        random_num_gen: np.random.RandomState instance
            Random number generator for reproducibility
        mutation_operator : str
            mutation operator, currently only supporting "inversion"
        target: str
            Target for fitness calculation.
            Set to "in" for in-distribution test data and "out" for out-of-distribution test data.
        """
        self.mutation_rate = 0.1
        self.penalty_weight = 200
        self.sum_of_distances = None
        self.penalty = None
        self.mutation_operator = mutation_operator
        self.target = target
        self.random_num_gen = random_num_gen
        self.chromosome = self.create_chromosome() if chromosome is None else chromosome
        self.fitness = self.calculate_fitness()

    def create_chromosome(self):
        """
        Creates a chromosome i.e. a random train-test split.
        It is an array with length equal to #points where 0 indicates training and 1 indicates test

        Returns
        -------
        np.ndarray
            Chromosome i.e. a random train-test split
        """
        num_points = len(points)
        point_idx = list(range(num_points))
        _, test_point_idx = train_test_split(point_idx, test_size=num_test_points, random_state=self.random_num_gen)
        chromosome = np.zeros(num_points, dtype=int)
        chromosome[test_point_idx] = 1
        return chromosome

    def calculate_fitness(self):
        """
        Calculates the fitness of the individual which is maximized.
        The fitness is defined as the sum of cosine distances between nearest neighboring test points.
        A penalty is subtracted from the fitness to force class-balance in the split.

        Returns
        -------
        float
            Fitness
        """
        # Get test point coordinates
        test_point_idx = np.nonzero(self.chromosome)
        train_point_idx = np.nonzero(self.chromosome == 0)
        test_points = points.iloc[test_point_idx]
        train_points = points.iloc[train_point_idx]

        preprocessor = setup_preprocessing_pipeline(points.columns)
        preprocessor.fit(train_points)
        test_points = preprocessor.transform(test_points)

        # ============================================

        if self.target == "in":
            nn = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="cosine", n_jobs=16).fit(test_points)
            distances, indices = nn.kneighbors(test_points)
            distances = distances[:, 1]  # remove distance of each point to itself

            # Compute sum of distances between nearest neighbors
            self.sum_of_distances = np.sum(distances)
        else:
            nn = NearestNeighbors(n_neighbors=test_points.shape[0], algorithm="brute", metric="cosine", n_jobs=16).fit(test_points)
            distances, indices = nn.kneighbors(test_points)
            distances = distances[:, 1:]

            # Compute sum of distances between all vectors
            self.sum_of_distances = np.sum(distances) * 0.001

        # Penalty: Difference between class distribution in test set and overall
        self.penalty = 0
        for i, _ in enumerate(class_names):
            test_point_labels = labels[i][test_point_idx]
            dist = test_point_labels.sum() / len(test_point_labels)
            self.penalty += abs(class_distributions[i] - dist)

        fitness = self.sum_of_distances - (self.penalty_weight * self.penalty)
        return fitness

    def _force_num_test_points(self, chromosome):
        num_diff = num_test_points - chromosome.sum()

        # not enough test points
        if num_diff > 0:
            train_idx = np.where(chromosome == 0)[0]
            random.shuffle(train_idx)
            for i in range(num_diff):
                chromosome[train_idx[i]] = 1

        # Too many test points
        elif num_diff < 0:
            test_idx = np.nonzero(chromosome)[0]
            random.shuffle(test_idx)
            for i in range(-num_diff):
                chromosome[test_idx[i]] = 0

        return chromosome

    def _mutation(self, chromosome):
        if self.mutation_operator == "inversion":
            num = random.randint(0, len(chromosome))
            start_point = random.randint(0, len(chromosome) - num)
            chromosome[start_point: start_point+num] = (chromosome[start_point: start_point+num] == 0).astype(int)
            chromosome = self._force_num_test_points(chromosome)
        return chromosome

    def crossover(self, parent2):
        """
        One-point crossover

        Parameters
        ----------
        parent2 : Individual
            Parent with which a crossover is performed

        Returns
        -------
            Individual
        Child of this Individual and the second Individual
        """
        # One point crossover
        crossover_point = random.randint(0, len(self.chromosome))

        # Crossover
        child_chromosome = self.chromosome.copy()
        child_chromosome[crossover_point:] = parent2.chromosome[crossover_point:]

        # Ensure that the number of ones in the chromosome == NUM_TEST_POINTS
        child_chromosome = self._force_num_test_points(child_chromosome)

        # Mutation
        if random.random() < self.mutation_rate:
            child_chromosome = self._mutation(child_chromosome)

        return Individual(child_chromosome, target=self.target)


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_operator, patience, target="in", log_name=None):
        """
        Genetic Algorithm for splitting multidimensional data into a training and a test dataset.
        It aims for a test dataset following the distribution of the overall data.

        Parameters
        ----------
        population_size : int
            Number of individuals in the population
        mutation_operator : str
            Name of mutation operator, currently only allows "inversion"
        patience : int
            Number of generations to wait for improvement before termination
        log_name : str
            Name for plots and log file
        target: str
            Target for fitness calculation.
            Set to "in" for in-distribution test data and "out" for out-of-distribution test data.
        """
        self.population_size = population_size
        self.mutation_operator = mutation_operator
        self.patience = patience
        self.target = target
        self.log_name = log_name

        self.population = None
        self.max_fitness_list = []
        self.median_fitness_list = []

    def create_initial_population(self):
        """
        Creates the initial population by iteratively creating Individuals.
        This function needs to be called before running the genetic algorithm.
        """
        print("Creating initial population...")
        population = []
        rng = np.random.RandomState(SEED)
        for _ in range(self.population_size):
            population.append(Individual(target=self.target, random_num_gen=rng))
        self.population = population
        print("Done.")

    def run(self):
        """
        Runs the genetic algorithm: Fitness evaluation, parent selection, crossover (with mutation)

        Returns
        -------
        Individual
            The fittest individual i.e. the final train-test split
        """
        generation = 0
        num_generations_without_improvement = 0

        while True:
            # ==== E v a l u a t e   f i t n e s s ====
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            fittest_individual = self.population[0]
            fitness_list = [self.population[i].fitness for i in range(self.population_size)]

            max_fitness = fittest_individual.fitness
            median_fitness = np.median(fitness_list)

            print(f"Gen {generation}:\tMedian fitness: {median_fitness:.4f}\tMax fitness: {max_fitness:.4f}")

            if self.log_name is not None:
                if generation % 10 == 0:
                    test_idx = np.nonzero(fittest_individual.chromosome)[0]
                    test_ids = df.iloc[test_idx].patient_id.tolist()
                    temp_df = labels_df.copy()
                    temp_df["dataset"] = temp_df.apply(lambda x: "test" if x["patient_id"] in test_ids else "training", axis=1)
                    # UMAP plot with colored training/test points
                    if not args.no_plots:
                        plot_umap_points(
                            dataset_split_df=temp_df,
                            title=None,
                            file_path=results_dir / f"{args.log_name}/dataset_split_umap_gen{generation}.png",
                            show=False
                        )

            if num_generations_without_improvement > self.patience:
                print(f"End: No improvement for {self.patience} generations")
                break

            if len(self.max_fitness_list) > 0:
                # Compare with previous best fitness
                if max_fitness <= self.max_fitness_list[-1]:
                    num_generations_without_improvement += 1
                else:
                    num_generations_without_improvement = 0

            new_generation = []

            # ========= S e l e c t i o n =========

            # Elitism
            num_elite = int(0.1 * self.population_size)
            new_generation.extend(self.population[:num_elite])

            # Tournament selection
            num_childs = self.population_size - num_elite
            tournament_size = 5
            selected_parents = []
            for _ in range(num_childs):
                tournament = random.sample(range(num_childs), tournament_size)
                tournament_scores = [fitness_list[i] for i in tournament]
                winner_idx = tournament[tournament_scores.index(max(tournament_scores))]
                selected_parents.append(self.population[winner_idx])

            # ========= C r o s s o v e r =========
            for i in range(0, num_childs, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i + 1]
                child1 = parent1.crossover(parent2)
                child2 = parent1.crossover(parent2)
                new_generation.append(child1)
                new_generation.append(child2)

            self.population = new_generation
            self.max_fitness_list.append(max_fitness)
            self.median_fitness_list.append(median_fitness)
            generation += 1

            # Log fitness of generations
            log = pd.DataFrame({
                "generation": list(range(generation)),
                "max_fitness": self.max_fitness_list,
                "median_fitness": self.median_fitness_list
            })
            log.to_csv(f"{results_dir}/{self.log_name}/log.csv", index=False)

            # Get current train-test split
            testidx = np.nonzero(self.population[0].chromosome)[0]
            testids = df.iloc[testidx].patient_id.tolist()
            labels_df["dataset"] = labels_df.apply(lambda x: "test" if x["patient_id"] in testids else "training", axis=1)
            # save intermediate result
            intermediate_json_path = results_dir / f"{args.log_name}/dataset_split_intermediate.json"
            labels_df[["patient_id", "dataset"]].to_json(intermediate_json_path, orient="records", indent=1)

        # Fittest individual of final generation
        fittest = self.population[0]

        return fittest


def plot_class_distribution(train_indices, test_indices, class_idx, file_path=None):
    df_train = pd.DataFrame({class_names[class_idx]: labels[class_idx][train_indices]})
    df_test = pd.DataFrame({class_names[class_idx]: labels[class_idx][test_indices]})

    # Count plot (class balance)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    cp = sns.countplot(data=df_train, x=class_names[class_idx], ax=ax1, palette="Set2")
    cp.bar_label(cp.containers[0])
    ax1.set_ylim([0, len(points)])
    ax1.set_title(f"Train (class dist: {sum(labels[class_idx][train_indices]) / len(labels[class_idx][train_indices]):.2f})")

    cp = sns.countplot(data=df_test, x=class_names[class_idx], ax=ax2, palette="Set2")
    cp.bar_label(cp.containers[0])
    ax2.set_ylim([0, len(points)])
    ax2.set_title(f"Test (class dist: {sum(labels[class_idx][test_indices]) / len(labels[class_idx][test_indices]):.2f})")
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()


def create_gif(plot_directory, gif_filepath):
    filenames = os.listdir(plot_directory)
    filenames = [f for f in filenames if "dataset_split_umap_gen" in f]
    filenames = natsorted(filenames)

    images = []
    for filename in filenames:
        images.append(io.imread(f"{plot_directory}/{filename}"))

    frame_durations = [0.1] * len(images)  # show first and last frame longer
    frame_durations[0] = 2
    frame_durations[-1] = 2
    io.mimsave(gif_filepath, images, duration=frame_durations)


def plot_umap_points(dataset_split_df, title=None, file_path=None, show=True):
    rcParams.update({"font.size": 8})
    rcParams["svg.fonttype"] = "none"
    df_umap = umap_embedding.merge(dataset_split_df, on="patient_id")
    palette = {"training": sns.color_palette("Set2")[0], "test": sns.color_palette("Set2")[1]}

    # Scatter plot (train / test points)
    plt.figure(figsize=(1.75, 1.75))
    ax = sns.scatterplot(data=df_umap, x="UMAP 1", y="UMAP 2", s=2.5, palette=palette, hue="dataset")

    ax.set_aspect("equal")
    plt.legend()
    sns.despine()
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=6)
    plt.xlabel("UMAP 1", fontsize=6)
    plt.ylabel("UMAP 2", fontsize=6)
    plt.tight_layout()

    if title is not None:
        plt.title(title)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_fitness_comparison(log_names, results_dir, log_labels, file_path=None):
    plt.figure(figsize=(4, 2))
    for log_name, label in zip(log_names, log_labels):
        subdirs = [d for d in os.listdir(results_dir) if log_name in d]
        df = []
        for subdir in subdirs:
            log_path = Path(results_dir/subdir)/"log.csv"
            if os.path.exists(log_path):
                df.append(pd.read_csv(log_path))
        df = pd.concat(df)
        sns.lineplot(df, x="generation", y="max_fitness", errorbar="sd", label=label)
    if len(log_names) > 1:
        plt.legend()
    plt.tight_layout()
    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_directory", type=str, help="Path to directory with extracted features (csv files)")
    parser.add_argument("results_directory", type=str, help="Path to directory where plots and results will be saved")
    parser.add_argument("log_name", type=str, help="Name of the GA run, a subdirectory with this name will be created")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in", dest="distribution", action="store_const", const="in", help="Split with in-distribution test data")
    group.add_argument("--out", dest="distribution", action="store_const", const="out", help="Split with out-of-distribution test data")
    parser.add_argument("--pop", dest="population_size", type=int, default=10000, help="Number of individuals in the population")
    parser.add_argument("--patience", dest="patience", type=int, default=50, help="Number of generations to wait for improvement before termination")
    parser.add_argument("--no_plots", action="store_true", help="Disable saving intermediate UMAP plots as png and gif")
    args = parser.parse_args()

    data_dir = Path(args.data_directory)
    results_dir = Path(args.results_directory)
    plt.switch_backend("agg")

    # Set random seed for reproducibility
    random.seed(SEED)

    # Test dataset size = 20% of all 763 patients
    total_num_patients = 763
    num_test_points = int(0.2 * total_num_patients)

    # Load data
    clinical = pd.read_csv(data_dir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(data_dir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(data_dir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density= pd.read_csv(data_dir/"tma_cell_density.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    # Convert to numpy arrays
    points = df.drop("patient_id", axis=1)

    # Load class labels
    class_names = ["recurrence", "survival_status"]
    labels_df = pd.read_csv(data_dir/"targets.csv", dtype={"patient_id": str})
    labels_temp = labels_df.copy()
    labels_temp = labels_temp.merge(df, on="patient_id", how="right").reset_index(drop=True)
    labels_temp = labels_temp.replace({"no": 0, "yes": 1, "living": 0, "deceased": 1})

    # Create lists of labels for each class of interest
    labels = []
    class_distributions = []
    for class_name in class_names:
        class_labels = labels_temp[class_name].tolist()
        labels.append(class_labels)
        class_distributions.append(np.sum(class_labels) / len(class_labels))
    labels = np.array(labels)

    # Get UMAP embedding (only for visualization)
    umap_embedding = get_umap_embedding(str(data_dir), umap_min_dist=0.2, umap_n_neighbors=15)

    # Create directory for results
    if os.path.isdir(results_dir/args.log_name):
         raise ValueError(f"Folder {results_dir}/{args.log_name}/ already exists.")
    else:
        os.mkdir(results_dir/args.log_name)

    # Run genetic algorithm to find points for the test dataset
    ga = GeneticAlgorithm(
        population_size=args.population_size,
        mutation_operator="inversion",
        target=args.distribution,
        patience=args.patience,
        log_name=args.log_name
    )
    ga.create_initial_population()
    solution = ga.run()

    # Get final train-test split
    test_idx = np.nonzero(solution.chromosome)[0]
    train_idx = np.where(solution.chromosome == 0)[0]
    test_ids = df.iloc[test_idx].patient_id.tolist()
    labels_df["dataset"] = labels_df.apply(lambda x: "test" if x["patient_id"] in test_ids else "training", axis=1)

    # save to .csv
    labels_df[["patient_id", "dataset"]].to_json(results_dir/f"dataset_split_{args.distribution}.json", orient="records", indent=1)

    # Class distributions
    plot_class_distribution(train_idx, test_idx, 0, f"{results_dir}/{args.log_name}/class_dist_recurrence.png")
    plot_class_distribution(train_idx, test_idx, 1, f"{results_dir}/{args.log_name}/class_dist_survival_status.png")

    # Fitness over time
    plot_fitness_comparison(
        log_names=[args.log_name],
        log_labels=[None],
        results_dir=results_dir,
        file_path=results_dir/f"{args.log_name}/fitness.png"
    )

    # UMAP plot with colored training/test points
    plot_umap_points(
        dataset_split_df=labels_df,
        title=None,
        file_path=results_dir/f"{args.log_name}/dataset_split_umap.svg",
        show=False
    )

    # Create gif from images showing UMAP with colored training/test plots
    if not args.no_plots:
        create_gif(plot_directory=results_dir/args.log_name, gif_filepath=Path(results_dir/args.log_name)/"dataset_split_umap.gif")
