### YOU! YES, YOU! Do you ever have this issue where you want to watch your favorite Japanese shows on a Japanese website that requires you to type things in Japanese, but all you have is a lousy US-layout keyboard? 

So, then you make your way to Yahoo! JAPAN, type in the Japanese using English characters (Romanji), hope the search results find what you're looking for, copy its Japanese characters, paste them into the Japanese TV-show website, and then finally start watching your favorite Japanese show. Then do it all over again for every new show you want to look up. Do you ever have that problem? No? Well, my Significant Other (SO) does.

### Have you ever thought to yourself "If only I could write the Japanese character myself, the computer recogizes what I want, and gives me back a copyable text!"?

That's exactly what I thought when I heard my SO's struggles. I also already had a drawing tablet sitting that was seeing little use. I had all of the physical technology, and a significant portion of the knowledge required to create this solution.

<br/>

--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------

# Current Milestone Progress

I want to implement this solution in a single Window's application, entirely written in python. There are a couple milestones that mark significant progress for this solution, some of which I need to learn new things before implementing:

| Milestone                         | Progress     |
| ----------------------------- 	| -------------:|
| Read drawing tablet data with Python            	| ▮▮▮▮▮▮▮▮▮▮ | 
| Create GUI visuals/assets  			| ▮▮▯▯▯▯▯▯▯▯ | 
| Create more training data with VAE          		| ▮▮▮▮▮▮▯▯▯▯ | 
| Create OCR model      | ▯▯▯▯▯▯▯▯▯▯ |
| Combine all into one application			| ▮▯▯▯▯▯▯▯▯▯ | 


<br/>

--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------

# Progress Details and Future Work

## A. Read drawing tablet data with Python
I struggle with this one for awhile. I eventually found some luck using the pyglet package, and after having some enlightening conversation on their discord, I was able to get it in working order.

My tablet is XP-PEN's Star 03 V2 Pen Tablet. Much of the struggle came from this tablet being fairly uncommon, so most python packages didn't recognize it as a tablet. 

Can read more details in the "app" subfolder.


## B. Create GUI visuals/assets
I will implementing this entirely in pyglet, so that the tablet and the widgets/app work well together. I have written some basic functionalilty into the app to confirm I could get the tablet to work the way I wanted it to and figure out what kind of data I can collect from it. However, I wanted to get the PyTorch models working before fidgeting around with GUI details.

Can read more details in the "app" subfolder.


## C. Create more training data with VAE
I have implemented CNNs in the past, but this was my first time implementing a VAE. I started by taking heavy inspiration from ***Deep Learning for Classical Japanese Literature*** by Clanuwat et al. ([2018](https://arxiv.org/abs/1812.01718)). Utilizing a VAE to make more variations of the training data considering I currently only have 1 "computer precise" example of every character, and the OCR will need to be able to recognize "handwritten"  characters (from a digital drawing tablet).

Can read more details in the "model" subfolder.


## D. Create OCR model
Currently no progress. Will be taking heavy inspiration from ***Deep Learning for Classical Japanese Literature*** by Clanuwat et al. ([2018](https://arxiv.org/abs/1812.01718)). 

Can read more details in the "model" subfolder.


## E. Combine all into one application
Entirely future work, there's not much need in doing this until everything else is pieced together. However, I should think about it a little bit so I don't have to completely reorganize all the project files in the future because I didn't think about some important aspect at the start.

Additionally, I know I'll need to figure out how to take the drawn characters and convert it into a data format that matches the OCR's input.


