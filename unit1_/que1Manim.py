from manim import *
import numpy as np

# Parameters
N = np.arange(1, 10, 1)  # Range of sample sizes
M = 10000  # Number of samples per distribution

# Generate data for each distribution
data_binom = [np.random.binomial(n=n, p=0.5, size=(n, M)) for n in N]
data_poiss = [np.random.poisson(5, size=(n, M)) for n in N]
data_norm = [np.random.normal(loc=0, scale=1, size=(n, M)) for n in N]
data_cauchy = [np.random.standard_cauchy(size=(n, M)) for n in N]
S = [data_binom, data_poiss, data_norm, data_cauchy]

dist_titles = ["Binomial", "Poisson", "Normal", "Cauchy"]

class DistributionAnimation(Scene):
    def construct(self):
        # Axes setup
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[0, 1.5, 0.5],
            axis_config={"include_numbers": True},
            x_axis_config={"numbers_to_include": np.arange(-5, 6, 1)},
            tips=False
        ).to_edge(DOWN)

        # Labels for axes
        x_label = axes.get_x_axis_label("X-axis")
        y_label = axes.get_y_axis_label("Density")

        # Title
        title = Text("Distribution Animation", font_size=40).to_edge(UP)

        # Add axes and labels to the scene
        self.play(Create(axes), Write(x_label), Write(y_label), Write(title))

        # Animation for each distribution and sample size
        for j, dist_data in enumerate(S):
            for i, n in enumerate(N):
                mean = np.mean(dist_data[i], axis=0)
                std_S = np.std(mean)

                # Create histogram data
                counts, bins = np.histogram(mean, bins=50, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # Histogram bars
                bars = VGroup(*[
                    Rectangle(
                        width=bins[1] - bins[0],
                        height=c,
                        stroke_width=0,
                        fill_opacity=0.75
                    ).move_to(axes.c2p(center, c / 2), aligned_edge=DOWN)
                    for center, c in zip(bin_centers, counts)
                ])

                # Gaussian fit curve
                gaussian_curve = axes.plot(
                    lambda x: (1 / (std_S * np.sqrt(2 * np.pi))) * np.exp(-((x - np.mean(mean)) ** 2) / (2 * std_S ** 2)),
                    x_range=[np.min(mean) - 1, np.max(mean) + 1],
                    color=RED
                )

                # Update title
                dist_title = Text(f"{dist_titles[j]} Distribution - N = {n}", font_size=24).next_to(axes, UP)

                # Show histogram and Gaussian fit
                self.play(
                    FadeIn(bars),
                    Create(gaussian_curve),
                    Transform(title, dist_title),
                    run_time=0.5
                )

                # Remove current plot before the next iteration
                self.play(FadeOut(bars), Uncreate(gaussian_curve), run_time=0.5)

        # End animation
        self.play(FadeOut(axes), FadeOut(x_label), FadeOut(y_label), FadeOut(title))
