import tkinter
class TextBox:
    def __init__(self, topX, topY, bottomX, bottomY, fill, outline, 
    outlineWidth, font, textColor, numberOfLines, numberOfLettersInALine, editable):
        self.topX = topX
        self.topY = topY
        self.bottomX = bottomX
        self.bottomY = bottomY
        self.fill = fill
        self.outline = outline
        self.numberOfLines = numberOfLines
        self.outlineWidth = outlineWidth
        self.font = font
        self.textColor = textColor
        self.numberOfLettersInALine = numberOfLettersInALine
        self.editble = editable

        self.listOfLinesInTextBox = ['']*self.numberOfLines
        self.currentLine = 0

    #draws the textbox
    def drawTextBox(self, canvas):
        canvas.create_rectangle(self.topX, self.topY, self.bottomX, 
        self.bottomY, fill=self.fill, outline=self.outline, 
        width = self.outlineWidth)
    #draws the words in the textbox
    def drawWords(self, canvas):
        for i in range (0,len(self.listOfLinesInTextBox)):
            canvas.create_text(self.topX+5, (self.topY+5)+i*20, anchor = 'nw',
                            text=self.listOfLinesInTextBox[i], 
                            font=self.font, fill=self.textColor)

    
    def controlV(self):
        #cheks if textbox is full or adds a new line
        i = self.currentLine
        if( len(self.listOfLinesInTextBox[i]) + 1 >= self.numberOfLettersInALine ):
            if( i + 1 >= len(self.listOfLinesInTextBox) ):
                print("Error: textBox is Full")
                return None
            else:
                self.currentLine = self.currentLine + 1
        #information from https://www.tcl.tk/man/tcl8.6/TkCmd/clipboard.html
        #gets info from clipboard
        tk = tkinter.Tk()###
        tk.withdraw()
        i = self.currentLine
        text = tk.clipboard_get() ###
        for word in text.split(" "):
            print(f"current word{word}")
            i = self.currentLine
            if len(self.listOfLinesInTextBox[i]) + len(word) >= self.numberOfLettersInALine:
                if( len(word) >= self.numberOfLettersInALine):
                    print("Error: Word in clipboard/chunk is longer then the line")
                    return None
                if( i + 1 >= len(self.listOfLinesInTextBox) ):
                    print("Error: textBox is Full")
                    return None
                else:
                    self.currentLine = self.currentLine + 1
            i = self.currentLine
            self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i] + word
            self.space()
        self.backspace
        
    #adds a chunkof text to the textbox
    def addChunkOfText(self, text):
        #checks if paste is too long or adds a new line

        for word in text.split(" "):
            self.addOneWord(word)
        self.backspace

    #adds one word to the textbox
    def addOneWord(self, word):
        i = self.currentLine
        if len(self.listOfLinesInTextBox[i]) + len(word) >= self.numberOfLettersInALine:
            if( len(word) >= self.numberOfLettersInALine):
                print("Error: Word in clipboard/chunk is longer then the line")
                return None
            if( i + 1 >= len(self.listOfLinesInTextBox) ):
                print("Error: textBox is Full")
                return None
            else:
                self.currentLine = self.currentLine + 1
        i = self.currentLine
        self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i] + word
        self.space()

    def addChunkOfTextWithoutSpace(self, text):
    #checks if paste is too long or adds a new line

        for word in text.split(" "):
            self.addOneWordWithoutSpace(word)
        self.backspace

    def addOneWordWithoutSpace(self, word):
        i = self.currentLine
        if len(self.listOfLinesInTextBox[i]) + len(word) >= self.numberOfLettersInALine:
            if( len(word) >= self.numberOfLettersInALine):
                print("Error: Word in clipboard/chunk is longer then the line")
                return None
            if( i + 1 >= len(self.listOfLinesInTextBox) ):
                print("Error: textBox is Full")
                return None
            else:
                self.currentLine = self.currentLine + 1
        i = self.currentLine
        self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i] + word
        # self.space()


    def space(self):
        #cheks if textbox is full or adds a new line
        i = self.currentLine
        if( len(self.listOfLinesInTextBox[i]) + 1 >= self.numberOfLettersInALine ):
            if( i + 1 >= len(self.listOfLinesInTextBox) ):
                print("Error: textBox is Full")
                return None
            else:
                self.currentLine = self.currentLine + 1
        #adds a space
        i = self.currentLine
        self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i] + " "

    def backspace(self):
        #removes last character
        i = self.currentLine
        length = len(self.listOfLinesInTextBox[i])
        if( len(self.listOfLinesInTextBox[i]) == 0):
            if ( i > 0 ):
                self.currentLine = self.currentLine -1
                i = self.currentLine
                self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i][:length-1]
            else:
                return None
        else:
            self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i][:length-1]
        

    def addRegLetter(self, key):
        #cheks if textbox is full or adds a new line
        i = self.currentLine
        if( len(self.listOfLinesInTextBox[i]) + 1 >= self.numberOfLettersInALine ):
            if( i + 1 >= len(self.listOfLinesInTextBox) ):
                print("Error: textBox is Full")
                return None
            else:
                self.currentLine = self.currentLine + 1
        #adds letter to the textbox
        i = self.currentLine
        self.listOfLinesInTextBox[i] = self.listOfLinesInTextBox[i] + key

    def replaceTextInTextBoxWithTextFromListOfStringsWithUpperHighlight(self, 
    ListOfSimplifiedWords, selectedWordIndex):
        self.listOfLinesInTextBox = ['']*self.numberOfLines
        self.currentLine = 0
        for i in range(0, len(ListOfSimplifiedWords)):
            if( i == selectedWordIndex ):
                self.addOneWordWithoutSpace(ListOfSimplifiedWords[i].upper())
            else:
                self.addOneWordWithoutSpace(ListOfSimplifiedWords[i])

    def clearTextInTextBox(self):
        self.listOfLinesInTextBox = ['']*self.numberOfLines
        self.currentLine = 0

    def getListOfLinesInTextBox(self):
        return self.listOfLinesInTextBox

    #returns demensions of the textbox
    def getDim(self):
        return self.topX, self.topY, self.bottomX, self.bottomY
    
    #returns number of lines in the textbox
    def getNumberOfLines(self):
        return self.numberOfLines