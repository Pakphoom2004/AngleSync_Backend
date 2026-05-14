import matplotlib.pyplot as plt

# Generate risk graph from frame-by-frame risk scores
def generate_risk_graph(
        risk_scores: tuple,
        highest_risk_frame_index: int,
        fps: int = 30
):

    time_axis = [
        round(frame / fps, 2)
        for frame in range(len(risk_scores))
    ]

    highest_risk_score = risk_scores[
        highest_risk_frame_index
    ]

    highest_risk_time = time_axis[
        highest_risk_frame_index
    ]

    plt.figure(figsize=(12, 6))

    plt.plot(
        time_axis,
        risk_scores,
        linewidth=2
    )

    plt.scatter(
        highest_risk_time,
        highest_risk_score,
        s=120
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Risk Score (%)")
    plt.title("Movement Risk Analysis")
    plt.grid(True)
    plt.tight_layout()
    output_path = "outputs/risk_graph.png"
    plt.savefig(output_path)
    plt.close()

    return output_path