# A2B.py
# This is the main Command Line Interface (CLI) entry point for the TBRGS application.
# It should accept positional arguments in the format:
#     python A2B.py <origin_node> <destination_node> <time> <model_type>
# Example:
#     python A2B.py 2000 2825 1100 LSTM

import sys

def main():
    # TODO: 
    # 1. Parse command-line arguments:
    #    - sys.argv[1]: Origin SCATS site ID (int/str)
    #    - sys.argv[2]: Destination SCATS site ID (int/str)
    #    - sys.argv[3]: Departure Time (e.g., "1100" for 11:00 AM)
    #    - sys.argv[4]: Model Selection (e.g., "LSTM" or "GRU")
    #
    # 2. Input Validation:
    #    - Verify that all arguments are provided.
    #    - Check if origin and destination exist in the road network.
    #    - Validate the time format (HHMM).
    #    - Validate the model selection (only LSTM or GRU allowed).
    #
    # 3. Load Model:
    #    - Load the trained model weights from the 'models_saved/' folder based on the model choice.
    #
    # 4. Predict Traffic Flow:
    #    - Predict the traffic flow on all links at the specified time step using the loaded model.
    #
    # 5. Convert Flow to Travel Time:
    #    - Run the predicted flow values through the flow-to-speed conversion formula.
    #    - Apply speed capping (limit speed to 60 km/h).
    #    - Compute travel times (seconds) for each segment.
    #
    # 6. Build Map Graph:
    #    - Update the road network graph's edge weights with the calculated travel times.
    #
    # 7. Run A* Search:
    #    - Run the custom A* search algorithm from origin to destination.
    #
    # 8. Print Results:
    #    - Print the fastest route path (list of nodes) and the estimated travel time.
    print("Welcome to TBRGS (Traffic-Based Route Guidance System)!")
    if len(sys.argv) >= 5:
        print(f"Routing from {sys.argv[1]} to {sys.argv[2]} at {sys.argv[3]} using {sys.argv[4]}...")
    else:
        print("Usage: python A2B.py <origin_node> <destination_node> <time> <model_type>")
    pass

if __name__ == "__main__":
    main()
