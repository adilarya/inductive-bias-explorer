run1 notes:
- CNN
    - train loss reduces smoothly to ~0.5
    - val loss reduces smoothly to ~0.6
    - train acc -> ~83%
    - val acc -> ~79%

- MLP 
    - train loss reduces to ~0.9
    - val loss plateaus ~1.2
    - train acc -> ~68%
    - val acc -> ~57%

Observations
- Learning speed: 
    - CNN reaches ~70% val acc in ~15 epochs
    - MLP takes ~60-70 epochs to approach ~55-57%
    - OVERALL: CNN converges faster
    - INTERPRETATION: Locality + weight sharing align better with image statistics, making the potimization landscape easier
- Generalization gap:
    - CNN gap: ~4%
    - MLP gap: ~11%
    - OVERALL: MLP overfits more despite worse validation performance
    - INTERPRETATION: MLP has weaker inductive bias for spatial structure, so it fits training data in a less reusable way.
- Asymptotic performance:
    - CNN saturates near ~79-80% val acc
    - MLP saturates near ~57-58% val acc
    - OVERALL: even with enough epochs, MLP cannot match the CNN
    - INTERPRETATION: it is not optimization failure, it is representation mismatch