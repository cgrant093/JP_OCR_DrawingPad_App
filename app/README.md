


The files in this folder are all based on the python package pyglet and have two purposes:

1. Connect my digital drawing tablet to the computer through python
2. Create the GUI objects

<br/>

# Current Progress

## A. Connected the drawing tablet
My tablet is XP-PEN's Star 03 V2 Pen Tablet. There is a drawing surface with several buttons on the tablet. It has an associated pen device device. Within pyglet's code, I was able to determine that I can collect data on the following functions:

1. Whether the pen is "in range" of the tablet's surface (within ~2-3 cm or ~1 inch).
2. Whether the pen is touching the tablet's surface
3. The amount of pressure being applied to the pen tip.
4. The (x, y) location of the pen on the tablet's surface.
5. Both the "polar" and "azimuthal" angles of the pen's orientation relative to the surface.
6. One of the two pen buttons on the barrel. 
    1. (The other is just considered "double-left click" and I haven't figured out how to get it to register in pyglet)

I have not found use for the pen angle orientation. However, the pressure one is useful for dynamically applying different line thicknesses. I needed to make a new pyglet shape subclass to get this to work.


## B. Created a basic GUI
Currently the GUI only has a small drawing window. Additionally, I have it so that the pen only creates "calligraphy" strokes inside a subset of the app window. I've set up an "undo" function using one of the pen's barrel buttons. I have also created one other app widget that's currently non-functional. 

<br/>

# Future Work

Now that I have the basics down, I plan on moving on to creating the model that will be able to recognize the Japanese Characters (both Kanji and Kana). However, here are some things I am thinking about.

1. Make a column of functional widget buttons that can either interact with the pen and/or interact with the column of buttons on the tablet?
2. Somehow create some functionality where the program gives you the top 5 character guesses and the user can choose which they wanted
    1. Should I try to add some sort of scroll mechanic so they can look further down the list?
    2. How should the guessing work? I'm thinking either the user presses a button when they are ready for it to guess, or it guesses after each newly added or deleted stroke?
3. There should be a little text box that displays and stores all of the current sessions chosen guesses as copyabale computer text characters
    1. Additionally, there should be some way to say "now copy this text" so the user can paste it elsewhere
    2. Probably should I also have some functionality for deleting all this text, so the user doesn't have to restart the app to start a new phrase
4. Should I improve the dynamic line thickness by incorporating the pen orientation angles? This could make it feel more like you're drawing calligraphy



