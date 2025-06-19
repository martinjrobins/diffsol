# Example: Fitting a spring-mass model to data

In this example we'll fit a damped spring-mass system to some synthetic data (using the model to generate the data). The system consists of a mass \\(m\\) attached to a spring with spring constant \\(k\\), and a damping force proportional to the velocity of the mass with damping coefficient \\(c\\).

\\[
\begin{align*}
\frac{dx}{dt} &= v \\\\
\frac{dv}{dt} &= -\frac{k}{m} x - \frac{c}{m} v
\end{align*}
\\]

where \\(v = \frac{dx}{dt}\\) is the velocity of the mass.

We'll use the argmin crate to perform the optimisation. To hold the synthetic data and the model, we'll create a `struct Problem` like so

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs::24}}
```

To use argmin we need to specify traits giving the loss function and its gradient. In this case we'll define a loss function equal to the sum of squares error between the model output and the synthetic data. 

$$
\text{loss} = \sum_i (y_i(p) - \hat{y}_i)^2
$$

where \\(y_i(p)\\) is the model output as a function of the parameters \\(p\\), and \\(\hat{y}_i\\) is the observed data at time index \\(i\\).

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:26:49}}
```

The gradient of this cost function with respect to the model outputs \\(y_i\\) is

$$
\frac{\partial \text{loss}}{\partial y_i} = 2 (y_i(p) - \hat{y}_i)
$$

We can calculate this using Diffsol's adjoint sensitivity analysis functionality. First we solve the forwards problem, generating a checkpointing struct. Using the forward solution we can then calculate \\(\frac{\partial loss}{\partial y_i}\\) for each time point, and then pass this into the adjoint backwards pass to calculate the gradient of the cost function with respect to the parameters.

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:51:81}}
```

In our main function we'll create the model, generate some synthetic data, and then call argmin to fit the model to the data.

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:83::}}
```

```
Mar 05 15:12:51.068 INFO L-BFGS
Mar 05 15:12:53.667 INFO iter: 0, cost: 6.099710479003417, best_cost: 6.099710479003417, gradient_count: 14, cost_count: 13, gamma: 1, time: 2.598805685
Mar 05 15:12:54.246 INFO iter: 1, cost: 1.9513388387772255, best_cost: 1.9513388387772255, gradient_count: 18, cost_count: 16, time: 0.579365776, gamma: 0.0009218600784406668
Mar 05 15:12:54.679 INFO iter: 2, cost: 1.1328003486802616, best_cost: 1.1328003486802616, gradient_count: 21, cost_count: 18, gamma: 0.0011158173475820988, time: 0.432772159
Mar 05 15:12:55.100 INFO iter: 3, cost: 0.36245408149937774, best_cost: 0.36245408149937774, gradient_count: 24, cost_count: 20, time: 0.421368339, gamma: 0.0010683972634626152
Mar 05 15:12:55.473 INFO iter: 4, cost: 0.005661451144141899, best_cost: 0.005661451144141899, gradient_count: 27, cost_count: 22, gamma: 0.0010337960155067532, time: 0.372749194
Mar 05 15:12:55.657 INFO iter: 5, cost: 0.0001534604284670027, best_cost: 0.0001534604284670027, gradient_count: 29, cost_count: 23, gamma: 0.0005519136139557582, time: 0.183717262
Mar 05 15:12:55.811 INFO iter: 6, cost: 0.000017178666309946563, best_cost: 0.000017178666309946563, gradient_count: 31, cost_count: 24, time: 0.154246656, gamma: 0.0005222593123731191
Mar 05 15:12:55.934 INFO iter: 7, cost: 0.0000011504081912133204, best_cost: 0.0000011504081912133204, gradient_count: 33, cost_count: 25, gamma: 0.0005839951848538406, time: 0.123233143
Mar 05 15:12:55.999 INFO iter: 8, cost: 0.0000000057304906811474396, best_cost: 0.0000000057304906811474396, gradient_count: 35, cost_count: 26, time: 0.064562531, gamma: 0.0004953781862470435
Mar 05 15:12:56.040 INFO iter: 9, cost: 0.00000000014827166483234088, best_cost: 0.00000000014827166483234088, gradient_count: 37, cost_count: 27, gamma: 0.0004739141598396127, time: 0.041539938
Mar 05 15:12:56.089 INFO iter: 10, cost: 0.00000000005665660355834637, best_cost: 0.00000000005665660355834637, gradient_count: 39, cost_count: 28, time: 0.048698991, gamma: 0.0006574550747061086
Mar 05 15:12:56.150 INFO iter: 11, cost: 0.00000000004046321552763034, best_cost: 0.00000000004046321552763034, gradient_count: 42, cost_count: 30, gamma: 0.0007768025353897974, time: 0.06118116
Mar 05 15:12:56.230 INFO iter: 12, cost: 0.000000000028544950162250156, best_cost: 0.000000000028544950162250156, gradient_count: 45, cost_count: 32, time: 0.079473263, gamma: 0.0007211431129919831
Mar 05 15:12:56.310 INFO iter: 13, cost: 0.000000000019824882126122364, best_cost: 0.000000000019824882126122364, gradient_count: 48, cost_count: 34, gamma: 0.0005276999999419798, time: 0.079908383
Mar 05 15:12:56.399 INFO iter: 14, cost: 0.000000000014773791668031016, best_cost: 0.000000000014773791668031016, gradient_count: 51, cost_count: 36, time: 0.088933521, gamma: 0.0006137808250157392
Mar 05 15:12:56.488 INFO iter: 15, cost: 0.000000000011918443921135866, best_cost: 0.000000000011918443921135866, gradient_count: 54, cost_count: 38, time: 0.088925265, gamma: 0.0006964726153446881
Mar 05 15:12:56.577 INFO iter: 16, cost: 0.000000000009847613120347547, best_cost: 0.000000000009847613120347547, gradient_count: 57, cost_count: 40, gamma: 0.0006788190120544423, time: 0.088873097
Mar 05 15:12:56.666 INFO iter: 17, cost: 0.000000000008250422213735562, best_cost: 0.000000000008250422213735562, gradient_count: 60, cost_count: 42, gamma: 0.0007147657772943421, time: 0.089253633
Mar 05 15:12:56.735 INFO iter: 18, cost: 0.000000000006963535307175679, best_cost: 0.000000000006963535307175679, gradient_count: 63, cost_count: 44, gamma: 0.0007522010953601424, time: 0.069181779
Mar 05 15:12:56.794 INFO iter: 19, cost: 0.000000000005843346318988485, best_cost: 0.000000000005843346318988485, gradient_count: 66, cost_count: 46, gamma: 0.0008115415177531896, time: 0.05931185
Mar 05 15:12:56.875 INFO iter: 20, cost: 0.000000000005668499496383206, best_cost: 0.000000000005668499496383206, gradient_count: 70, cost_count: 49, gamma: 0.0007981283294910353, time: 0.080732485
Mar 05 15:12:56.966 INFO iter: 21, cost: 0.000000000005235587953062947, best_cost: 0.000000000005235587953062947, gradient_count: 74, cost_count: 52, gamma: 0.0007824334700764565, time: 0.090509819
Mar 05 15:12:57.025 INFO iter: 22, cost: 0.000000000005176697246946799, best_cost: 0.000000000005176697246946799, gradient_count: 77, cost_count: 54, time: 0.05927997, gamma: 0.0007160320701376257
Mar 05 15:12:57.084 INFO iter: 23, cost: 0.000000000005123431411964367, best_cost: 0.000000000005123431411964367, gradient_count: 80, cost_count: 56, time: 0.059282524, gamma: 0.0006691254860074899
Mar 05 15:12:57.145 INFO iter: 24, cost: 0.000000000004888671439469577, best_cost: 0.000000000004888671439469577, gradient_count: 83, cost_count: 58, time: 0.060525746, gamma: 0.0006870970887038928
Mar 05 15:12:57.205 INFO iter: 25, cost: 0.000000000004640183634642808, best_cost: 0.000000000004640183634642808, gradient_count: 86, cost_count: 60, time: 0.060494317, gamma: 0.0006938418390339419
Mar 05 15:12:57.266 INFO iter: 26, cost: 0.000000000004202140058725012, best_cost: 0.000000000004202140058725012, gradient_count: 89, cost_count: 62, gamma: 0.0008074272439861259, time: 0.060721945
Mar 05 15:12:57.327 INFO iter: 27, cost: 0.000000000003819065447066373, best_cost: 0.000000000003819065447066373, gradient_count: 92, cost_count: 64, time: 0.060539541, gamma: 0.000745089693258659
Mar 05 15:12:57.388 INFO iter: 28, cost: 0.0000000000034723813904317016, best_cost: 0.0000000000034723813904317016, gradient_count: 95, cost_count: 66, time: 0.061195346, gamma: 0.0006679419374299279
Mar 05 15:12:57.449 INFO iter: 29, cost: 0.0000000000029793598830196132, best_cost: 0.0000000000029793598830196132, gradient_count: 98, cost_count: 68, gamma: 0.0006803520819495024, time: 0.061207339
Mar 05 15:12:57.510 INFO iter: 30, cost: 0.000000000002000300316746391, best_cost: 0.000000000002000300316746391, gradient_count: 101, cost_count: 70, gamma: 0.0007707919521852348, time: 0.061103644
Mar 05 15:12:57.571 INFO iter: 31, cost: 0.0000000000013081231359178465, best_cost: 0.0000000000013081231359178465, gradient_count: 104, cost_count: 72, gamma: 0.0007594375561408552, time: 0.061080561
Mar 05 15:12:57.632 INFO iter: 32, cost: 0.0000000000009129669554476711, best_cost: 0.0000000000009129669554476711, gradient_count: 107, cost_count: 74, gamma: 0.0007714623407119796, time: 0.061010138
Mar 05 15:12:57.693 INFO iter: 33, cost: 0.0000000000006624988392852611, best_cost: 0.0000000000006624988392852611, gradient_count: 110, cost_count: 76, time: 0.060805933, gamma: 0.0006929806628651383
Mar 05 15:12:57.732 INFO iter: 34, cost: 0.0000000000006457742400885126, best_cost: 0.0000000000006457742400885126, gradient_count: 112, cost_count: 77, gamma: 0.0006611075118236918, time: 0.038888894
Mar 05 15:12:57.771 INFO iter: 35, cost: 0.0000000000004844629418810476, best_cost: 0.0000000000004844629418810476, gradient_count: 114, cost_count: 78, time: 0.038867624, gamma: 0.0006746920875937051
Mar 05 15:12:57.832 INFO iter: 36, cost: 0.00000000000021331454699420582, best_cost: 0.00000000000021331454699420582, gradient_count: 117, cost_count: 80, time: 0.060733748, gamma: 0.0006901151177651382
Mar 05 15:12:57.870 INFO iter: 37, cost: 0.0000000000001604124255790092, best_cost: 0.0000000000001604124255790092, gradient_count: 119, cost_count: 81, gamma: 0.0007678508731651804, time: 0.038720877
Mar 05 15:12:57.909 INFO iter: 38, cost: 0.00000000000006736683702894452, best_cost: 0.00000000000006736683702894452, gradient_count: 121, cost_count: 82, time: 0.038724956, gamma: 0.0007450395288400451
Mar 05 15:12:57.948 INFO iter: 39, cost: 0.000000000000033549027916251836, best_cost: 0.000000000000033549027916251836, gradient_count: 123, cost_count: 83, gamma: 0.0007953280906606892, time: 0.038636279
Mar 05 15:12:57.986 INFO iter: 40, cost: 0.0000000000000009190438168193556, best_cost: 0.0000000000000009190438168193556, gradient_count: 125, cost_count: 84, gamma: 0.0008377658208246105, time: 0.038537432
Mar 05 15:12:58.024 INFO iter: 41, cost: 0.0000000000000000010640888952026633, best_cost: 0.0000000000000000010640888952026633, gradient_count: 127, cost_count: 85, gamma: 0.0008326080674532384, time: 0.037460313
Mar 05 15:12:58.060 INFO iter: 42, cost: 0.0000000000000000000003834750428076903, best_cost: 0.0000000000000000000003834750428076903, gradient_count: 129, cost_count: 86, time: 0.036246197, gamma: 0.0008348526737634932
OptimizationResult:
    Solver:        L-BFGS
    param (best):  [0.9999999999993618, 0.0999999999999094]
    cost (best):   0.0000000000000000000003834750428076903
    iters (best):  42
    iters (total): 43
    termination:   Solver converged
    time:          7.139444823s

Best parameter vector: [0.9999999999993618, 0.0999999999999094]
True parameter vector: [1.0, 0.1]
```