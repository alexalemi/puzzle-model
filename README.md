
I'd like to try to build a model of speed puzzling.  I want to try to get a sense of how speed puzzling times should vary with puzzle size and type.  Eventually it might be cool to try to use an image model to try to predict the difficulty of an individual puzzle.

The first problem is trying to get my hands on some puzzle data.  There are two sites that come to mind, first speedpuzzling.com which as the results from various competitions, though they are stored in a series of pdfs.

There is also myspeedpuzzling.com which has a whole bunch of user submitted data, but the site doesn't seem to be particularly easy to navigate.  In both cases I probably have to write some code to scrape or digitize the data.

Let's start with the data on speedpuzzling.com.

There are a series of results pages: 

 * https://www.speedpuzzling.com/results.html
 * https://www.speedpuzzling.com/results2024.html
 * https://www.speedpuzzling.com/results-2023.html
 * https://www.speedpuzzling.com/results-2022.html
 * https://www.speedpuzzling.com/results-2021.html
 * https://www.speedpuzzling.com/results-2020.html

There is also the USA jigsaw puzzle association results:

 * https://www.usajigsaw.org/2025-nationals-results
 * https://www.usajigsaw.org/2024-nationals-1
 * https://www.usajigsaw.org/2022-nationals

Also

 * https://www.webscorer.com/usajpa



## Model


In terms of modelling the puzzles, I feel like there are different aspects of puzzling that could contribute different terms.

 * A linear term ($N$) due to the number of pieces
 * A sqrt ($\sqrt N$) term for the size of the border.
 * A search like term ($N \log N$)
 * Some worst case quadratic term ($N^2$).

It feels like the place to start is to see how much I can explain puzzle solve times for different sized puzzles with these factors.

I also think there should be some kind of central elo like model, maybe broken down along these dimensions, where each puzzle has a "difficulty" and each solver has some kind of latent "ability".

