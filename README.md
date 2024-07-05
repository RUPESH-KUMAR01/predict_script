# Model Inference Time Comparison

This document compares the inference times of two different implementations for a series of images. The first set of data represents inference times using Ultralytics code, and the second set represents inference times using our own implementation.

## Inference Time Comparison

### Summary Table

| Image Number | Path                                                            | Ultralytics Inference Time (ms) | Our Code Inference Time (ms) |
|--------------|-----------------------------------------------------------------|---------------------------------|------------------------------|
| 1            | C:\Users\thata\intern\code\pre-built-models\assets\000000000009.jpg | 90.2                            | 92.37                        |
| 2            | C:\Users\thata\intern\code\pre-built-models\assets\000000000025.jpg | 80.1                            | 89.04                        |
| 3            | C:\Users\thata\intern\code\pre-built-models\assets\000000000030.jpg | 10.0                            | 5.00                         |
| 4            | C:\Users\thata\intern\code\pre-built-models\assets\000000000034.jpg | 10.0                            | 5.27                         |
| 5            | C:\Users\thata\intern\code\pre-built-models\assets\000000000036.jpg | 90.1                            | 80.48                        |
| 6            | C:\Users\thata\intern\code\pre-built-models\assets\000000000042.jpg | 10.0                            | 6.00                         |
| 7            | C:\Users\thata\intern\code\pre-built-models\assets\000000000049.jpg | 10.0                            | 6.95                         |
| 8            | C:\Users\thata\intern\code\pre-built-models\assets\000000000061.jpg | 80.2                            | 82.79                        |
| 9            | C:\Users\thata\intern\code\pre-built-models\assets\bus.jpg        | 80.0                            | 86.33                        |
| 10           | C:\Users\thata\intern\code\pre-built-models\assets\zidane.jpg     | 80.2                            | 89.60                        |

### Total Inference Times

- **Ultralytics Code Total Inference Time:** 3.32 seconds
- **Our Code Total Inference Time:** 2.09 seconds

## Speed Metrics (Our Code)

- **Preprocessing Time:** 1.0 ms per image
- **Inference Time:** 54.1 ms per image
- **Postprocessing Time:** 4.0 ms per image

**Total Average Processing Time per Image:** 59.1 ms

## Comparison

1. **Individual Inference Times:**
   - Inference times for individual images vary between the two implementations. Our code generally shows lower inference times for some images compared to Ultralytics.

2. **Total Inference Time:**
   - Our code demonstrates a faster total inference time (2.09 seconds) compared to the Ultralytics code (3.32 seconds).

3. **Complexity and Overhead:**
   - The Ultralytics code includes a complex saving method that expects a specific module structure, contributing to additional overhead time.
   - Our code avoids this complexity by not requiring the entire module structure, resulting in reduced overhead and faster inference times.
