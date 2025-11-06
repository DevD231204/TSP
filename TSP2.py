import tkinter as tk
from tkinter import Button, Canvas, Label
import math
import random

class TSPGUI:
    def __init__(self, master):
        self.master = master
        master.title("TSP Solver - Genetic Algorithm")
        master.configure(bg="#262626")

        # Info label
        self.info_label = Label(master, text="Click to add towns, then press Solve", bg="#262626", fg="white", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # Canvas setup
        self.canvas = Canvas(master, width=900, height=600, bg="#262626", highlightthickness=0)
        self.canvas.pack()

        # Solve button
        self.solve_btn = Button(master, text="Solve", command=self.start_solve, font=("Arial", 14), bg="#444", fg="white")
        self.solve_btn.pack(pady=10)

        # Data storage
        self.points = []
        self.best_path = []
        self.best_distance = float('inf')
        self.solving = False

        # GA parameters
        self.population_size = 150
        self.generations = 0
        self.max_generations = 1500
        self.population = []

        # Bind click
        self.canvas.bind("<Button-1>", self.add_point)
        self.redraw()

    def add_point(self, event):
        if not self.solving:
            self.points.append((event.x, event.y))
            self.best_path = []
            self.best_distance = float('inf')
            self.redraw()
            self.max_generations += self.max_generations//10
            self.population_size += self.population_size//10

    def redraw(self, current_path=None, current_dist=None, generation=None):
        self.canvas.delete("all")
        
        # Draw current best path (light gray)
        if current_path and len(current_path) > 1:
            for i in range(len(current_path)):
                x1, y1 = self.points[current_path[i]]
                x2, y2 = self.points[current_path[(i+1)%len(current_path)]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#FFFFFF", width=2)
        
        # Draw global best path (green)
        if self.best_path:
            for i in range(len(self.best_path)):
                x1, y1 = self.points[self.best_path[i]]
                x2, y2 = self.points[self.best_path[(i+1)%len(self.best_path)]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#00ff00", width=3)
        
        # Draw points
        for x, y in self.points:
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
        
        # Update info label
        if generation is not None and current_dist is not None:
            info_text = f"Generation: {generation}/{self.max_generations} | Current: {current_dist:.2f} km | Best: {self.best_distance:.2f} km"
            self.info_label.config(text=info_text)
    
    def calc_distance(self, path):
        """Calculate total distance of a path"""
        n = len(path)
        total = 0
        for i in range(n):
            x1, y1 = self.points[path[i]]
            x2, y2 = self.points[path[(i+1)%n]]
            total += math.hypot(x2-x1, y2-y1)
        return total

    def create_individual(self):
        """Create a random tour"""
        individual = list(range(len(self.points)))
        random.shuffle(individual)
        return individual

    def create_population(self, size):
        """Create initial population"""
        return [self.create_individual() for _ in range(size)]

    def fitness(self, individual):
        """Fitness = 1 / distance (higher is better)"""
        distance = self.calc_distance(individual)
        return 1 / (distance + 1)  # +1 to avoid division by zero

    def selection(self, population, fitness_scores):
        """Roulette wheel selection"""
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        
        selected = []
        for _ in range(len(population)):
            r = random.random()
            cumsum = 0
            for i, prob in enumerate(probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(population[i][:])
                    break
        return selected

    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX) for TSP"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                if pointer == size:
                    pointer = 0
                child[pointer] = city
                pointer += 1
        
        return child

    def mutate(self, individual, mutation_rate=0.02):
        """Swap mutation"""
        for _ in range(len(individual)):
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def evolve_population(self, population):
        """Create next generation"""
        fitness_scores = [self.fitness(ind) for ind in population]
        
        # Elitism: keep best individual
        best_idx = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_idx][:]
        
        # Selection
        selected = self.selection(population, fitness_scores)
        
        # Crossover and Mutation
        new_population = [best_individual]
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(selected, 2)
            child = self.order_crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        return new_population[:len(population)]

    def start_solve(self):
        n = len(self.points)
        if n < 2:
            self.info_label.config(text="Need at least 2 towns!")
            return
        
        if self.solving:
            return
        
        self.solving = True
        self.solve_btn.config(state="disabled")
        self.best_distance = float('inf')
        self.best_path = []
        
        # Initialize population
        self.population = self.create_population(self.population_size)
        self.generations = 0
        
        # Start GA evolution
        self.evolve_next()
    
    def evolve_next(self):
        if self.generations >= self.max_generations:
            # Finished
            self.solving = False
            self.solve_btn.config(state="normal")
            self.info_label.config(text=f"Solved! Best distance: {self.best_distance:.2f} km | Generations: {self.generations}")
            self.redraw()
            return
        
        # Evaluate current population
        fitness_scores = [self.fitness(ind) for ind in self.population]
        best_in_gen_idx = fitness_scores.index(max(fitness_scores))
        best_in_gen = self.population[best_in_gen_idx]
        dist_in_gen = self.calc_distance(best_in_gen)
        
        # Update global best
        if dist_in_gen < self.best_distance:
            self.best_distance = dist_in_gen
            self.best_path = best_in_gen[:]
        
        # Draw current generation
        self.redraw(best_in_gen, dist_in_gen, self.generations + 1)
        
        # Evolve to next generation
        self.population = self.evolve_population(self.population)
        self.generations += 1
        
        # Schedule next generation (500ms delay)
        self.master.after(5, self.evolve_next)

if __name__ == "__main__":
    root = tk.Tk()
    gui = TSPGUI(root)
    root.mainloop()
