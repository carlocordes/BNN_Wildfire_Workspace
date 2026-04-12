# Self-attention

A mechanism allowing a network to capture relationships between any two parts of the image, regardless of the distance between them.

    "How relevant is the information in the other patch (Key) to what I am currently looking for (Query)?"

Serves a scoring mechanism of every one token (patch) with another.

Calculation is done via 3 parameter vectors: [[query]], [[key]], [[value]]

