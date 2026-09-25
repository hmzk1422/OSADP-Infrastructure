import pyarrow.parquet as pq
import numpy as np


def analyze_event_times(path):
    parquet = pq.ParquetFile(path)

    previous_event_time = None
    unique_events = 0
    gaps = []

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["event_time"],
    ):
        event_times = batch.column("event_time").to_pylist()

        for event_time in event_times:
            # Ignore repeated book rows with the same event_time
            if (
                previous_event_time is not None
                and event_time == previous_event_time
            ):
                continue

            if previous_event_time is not None:
                gaps.append(
                    event_time - previous_event_time
                )

            previous_event_time = event_time
            unique_events += 1

    if gaps:
        gaps_array = np.array(gaps, dtype=np.float64)

        print("\nGap statistics:")
        print(f"Minimum:  {np.min(gaps_array):.3f} ms")
        print(f"1%:       {np.percentile(gaps_array, 1):.3f} ms")
        print(f"5%:       {np.percentile(gaps_array, 5):.3f} ms")
        print(f"Median:   {np.percentile(gaps_array, 50):.3f} ms")
        print(f"95%:      {np.percentile(gaps_array, 95):.3f} ms")
        print(f"99%:      {np.percentile(gaps_array, 99):.3f} ms")
        print(f"Maximum:  {np.max(gaps_array):.3f} ms")

        print(
            f"Average:  {np.mean(gaps_array):.3f} ms"
        )

    print(f"\nFile: {path}")
    print(f"Unique event times: {unique_events}")

    if gaps:
        average_gap_ms = np.mean(gaps_array)

        print(
            f"Average unique-event gap: "
            f"{average_gap_ms:.6f} ms"
        )

        print(
            f"Average unique-event gap: "
            f"{average_gap_ms / 1000:.6f} seconds"
        )


analyze_event_times(
    "05_every_event.parquet"
)