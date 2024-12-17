@echo off
set /a c=112900

echo %c%

for %%n in (100, 200, 400) do (
    for %%b in (0, 0.5, 1.0, 1.5, 2.0) do (
        for %%e in ("Normal", "Cauchy", "LogNormal") do (
            python dispersion_experiment.py ^
                -n %%n ^
                -p 6 ^
                -B 99 ^
                --trials 100 ^
                --beta %%b ^
                --epsilon-distribution %%~e ^
                --seed %c%
            set /a c+=1
        )
    )
)