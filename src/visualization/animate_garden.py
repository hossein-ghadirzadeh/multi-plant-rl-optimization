import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random


def animate_garden(env, q_early, q_late, pg_early, pg_late,
                   q_early_stats, q_late_stats, pg_early_stats, pg_late_stats,
                   save_path=None, fps=5):
   # create figure and 2x2 grid for animations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#e8f7ff")

    titles = [   # titles for each subplot with growth stats
        f"Q-Learning Early — AvgLen{q_early_stats[1]:.2f} | Alive{q_early_stats[2]*100:.0f}%",
        f"Q-Learning Late — AvgLen{q_late_stats[1]:.2f} | Alive{q_late_stats[2]*100:.0f}%",
        f"PG Early — AvgLen{pg_early_stats[1]:.2f} | Alive{pg_early_stats[2]*100:.0f}%",
        f"PG Late — AvgLen{pg_late_stats[1]:.2f} |Alive{pg_late_stats[2]*100:.0f}%"
    ]

    datasets = [q_early, q_late, pg_early, pg_late] # data for each panel
    x_spacing, y_base = 1.2, 0.4  # spacing for plants
    flower_colors = ["gold", "violet", "deeppink", "orange", "lightcoral"]  

    plots, flowers, texts, grounds = [], [], [], []
    for ax, title in zip(axes.flat, titles):
        ax.set_xlim(0, env.num_plants * 1.3)
        ax.set_ylim(0, env.max_height + 3)
        ax.set_facecolor("#d3f9d8")
        ground_patch = ax.add_patch(plt.Rectangle(
            (0, 0), env.num_plants * x_spacing, y_base, color="saddlebrown"))
        ax.set_title(title, fontsize=11, weight="bold", color="#2c3e50")
        ax.axis("off")

      # create plant stems and flower markers
        stems, blooms = [], []
        for i in range(env.num_plants):
            s, = ax.plot([], [], lw=3, color="darkgreen", alpha=0.9)
            f, = ax.plot([], [], marker="o",
                         color=random.choice(flower_colors),
                         markersize=0, markeredgecolor="black", markeredgewidth=0.5)
            stems.append(s)
            blooms.append(f)

        txt = ax.text(0.05, env.max_height + 2.3, "", fontsize=9,
                      color="#34495e", weight="bold")
        plots.append(stems)
        flowers.append(blooms)
        texts.append(txt)
        grounds.append(ground_patch)

    max_frames = max(len(d) for d in datasets)
    current_frame = [0]
    paused = [False]

   # initialize the animation 
    def init():
        elements = []
        for s_list, f_list, t in zip(plots, flowers, texts):
            for s, f in zip(s_list, f_list):
                s.set_data([], [])
                f.set_data([], [])
            t.set_text("")
            elements += s_list + f_list + [t]
        return elements
    
    # draw each frame of the animation
    def draw_frame(frame):
        elements = []
        shimmer = 0.01 * np.sin(frame / 8.0)
        for (data, s_list, f_list, t, ground) in zip(datasets, plots, flowers, texts, grounds):
            step = min(frame, len(data) - 1)
            heights = data[step]
            # update stem height and color based on plant growth
            for i, h in enumerate(heights):
                x = 0.6 + i * x_spacing
                s_list[i].set_data([x, x], [y_base, y_base + h])
                green_level = min(1.0, max(0.1, h / env.max_height))
                color = (0, green_level, 0)
                s_list[i].set_color(color)

                if h >= env.max_height * 0.8:
                    f_list[i].set_data([x], [y_base + h + 0.4])
                    f_list[i].set_markersize(9 + 2 * np.sin(frame / 5 + i))
                else:
                    f_list[i].set_data([], [])
                    f_list[i].set_markersize(0)

            ground.set_y(shimmer)
            alive = sum(h > 0.5 for h in heights)
            avg_len = np.mean(heights)
            t.set_text(
                f"Step {step+1}/{max_frames} — Alive {alive}/{env.num_plants} | Avg {avg_len:.2f}")
            elements += s_list + f_list + [t, ground]
        return elements
    
   # update frame 
    def update(_):
        if not paused[0]:
            current_frame[0] = (current_frame[0] + 1) % max_frames
        return draw_frame(current_frame[0])

    # ---------- Keyboard controls ----------
    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]
            print("Paused" if paused[0] else "Resumed")
        elif event.key == "right":
            paused[0] = True
            current_frame[0] = min(current_frame[0] + 1, max_frames - 1)
            print(f"Frame {current_frame[0]+1}")
            draw_frame(current_frame[0])
            fig.canvas.draw_idle()
        elif event.key == "left":
            paused[0] = True
            current_frame[0] = max(current_frame[0] - 1, 0)
            print(f"Frame {current_frame[0]+1}")
            draw_frame(current_frame[0])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    # create animation
    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=max_frames, interval=300, blit=True, repeat=False
    )

    # ---------- Optional Save ----------
    if save_path:
        print(f" Saving animation to: {save_path} ...")
        try:
            if save_path.lower().endswith(".gif"):
                ani.save(save_path, writer="pillow", fps=fps)
            elif save_path.lower().endswith((".mp4", ".mov", ".avi")):
                ani.save(save_path, writer="ffmpeg", fps=fps)
            else:
                print("unsupported file format. Use .gif or .mp4")
                return
            print("animation saved successfully!")
        except Exception as e:
            print(f"error during saving: {e}")

    plt.tight_layout()
    plt.show()
