from manim import *
import pandas as pd
import numpy as np

class CostoMarginalAnim(Scene):
    def construct(self):
        # Load and prepare data
        df = pd.read_excel("costos-marginales-online.xlsx", sheet_name="amCharts")
        df['item'] = pd.to_datetime(df['item'])

        x_vals = list(range(len(df)))
        y_vals = df['Crucero'].to_numpy()

        # Normalize y values to fit in Manim space (-3 to 3)
        y_scaled = np.interp(y_vals, (min(y_vals), max(y_vals)), [-3, 3])

        # Create Axes
        axes = Axes(
            x_range=[0, len(x_vals)-1, 4],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=6,
            axis_config={"include_numbers": False},
            tips=False,
        )

        # Gridlines
        grid = axes.get_grid()
        self.add(grid)

        # X-axis labels: real hours
        hours = df['item'].dt.strftime('%H:%M')
        for i in range(0, len(x_vals), 4):
            label = Text(hours[i], font_size=24).scale(0.4)
            label.next_to(axes.c2p(x_vals[i], -3), DOWN, buff=0.15)
            self.add(label)

        # Y-axis labels
        for y_tick in [-2, 0, 2]:
            val = np.interp(y_tick, [-3, 3], [min(y_vals), max(y_vals)])
            label = Text(f"{val:.0f}", font_size=24).scale(0.4)
            label.next_to(axes.c2p(0, y_tick), LEFT, buff=0.15)
            self.add(label)

        self.play(Create(axes))

        # Animate the progressive line
        graph_points = [axes.c2p(x, y) for x, y in zip(x_vals, y_scaled)]
        line = VMobject(color=ORANGE).set_points_as_corners([graph_points[0]])
        self.add(line)

        for i in range(1, len(graph_points)):
            new_line = VMobject(color=ORANGE).set_points_as_corners(graph_points[:i+1])
            self.play(Transform(line, new_line), run_time=0.05)

        self.wait(0.5)

        # Add Buy shaded region (index 10 to 15)
        buy_rect = axes.get_area(
            lambda x: 3,
            x_range=(10, 15),
            color=GRAY,
            opacity=0.3
        )
        buy_text = Text("Buy", font_size=32).move_to(axes.c2p(12.5, 2.4))
        self.play(FadeIn(buy_rect), FadeIn(buy_text))

        # Add Sell shaded region (index 21 to 23)
        sell_rect = axes.get_area(
            lambda x: 3,
            x_range=(21, 23),
            color=GREEN,
            opacity=0.3
        )
        sell_text = Text("Sell", font_size=32).move_to(axes.c2p(22, 2.4))
        self.play(FadeIn(sell_rect), FadeIn(sell_text))

        self.wait(2)
