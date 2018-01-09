# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from searchAgents import mazeDistance, ApproximateSearchAgent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if(successorGameState.isWin()):
          return float("inf");

        evalVal = 0
        
        foodList = newFood.asList()
        nearestFood = util.manhattanDistance(foodList[0], newPos)
        for foodPos in foodList:
          foodDist = util.manhattanDistance(foodPos, newPos)
          if(foodDist<nearestFood):
            nearestFood = foodDist

        evalVal = evalVal-nearestFood #further the nearest food, lower the evaluation value      

        # distance to ghost, since max manhattan distance to ghost can be 2
        for x in range(1, currentGameState.getNumAgents() - 1):
          ghostPacDiff = util.manhattanDistance(currentGameState.getGhostPosition(x), newPos)
          if(ghostPacDiff<3):
            evalVal = evalVal + ghostPacDiff
          else:
            evalVal = evalVal + 3       
        
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
          evalVal = evalVal + 100

        evalVal = evalVal + successorGameState.getScore()

        return evalVal

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # assume pac man is maximizer
        # ghosts are minimizers
        # assume pacman starts first
        def getMiniMaxAction(gameState, agentIndex, depth):
          
          curScore = float("inf")
          curAction = None

          if(agentIndex == 0):
            curScore = -float("inf")

          actions = gameState.getLegalActions(agentIndex)
          if(gameState.isWin() or gameState.isLose() or len(actions) == 0 or (agentIndex == 0 and self.depth == depth)):
            return (self.evaluationFunction(gameState), None)
          for action in actions:
            nextState = gameState.generateSuccessor(agentIndex, action)
            if(agentIndex == gameState.getNumAgents()-1):
              nextScore = getMiniMaxAction(nextState, 0, depth+1)[0]
            else:
              nextScore = getMiniMaxAction(nextState, agentIndex+1, depth)[0]
            if(agentIndex == 0):
              if(nextScore>curScore):
                curScore = nextScore
                curAction = action
            else:
              if(nextScore<curScore):
                curScore = nextScore
                curAction = action
          return (curScore, curAction)

        return getMiniMaxAction(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, alpha, beta, depth):          
          tmpScore = nextScore = -float("inf")
          nextAction = None
          actions = state.getLegalActions(0)
          if(state.isWin() or state.isLose() or len(actions) == 0 or self.depth == depth):
            return (self.evaluationFunction(state), None)
          for action in actions:
            if(alpha>beta):
              return (nextScore, nextAction)
            nextState = state.generateSuccessor(0, action)
            tmpScore = min_value(nextState, alpha, beta, 1, depth)[0]
            nextScore = max(tmpScore, nextScore)
            if(tmpScore==nextScore):              
              nextAction = action            
            alpha = max(alpha, nextScore)
          return (nextScore, nextAction)

        def min_value(state, alpha, beta, agentIndex, depth):
          tmpScore = nextScore = float("inf")
          nextAction = None
          actions = state.getLegalActions(agentIndex)
          if(state.isWin() or state.isLose() or len(actions) == 0):
            return (self.evaluationFunction(state), None)
          for action in actions:
            if(alpha>beta):
              return (nextScore, nextAction)
            nextState = state.generateSuccessor(agentIndex, action)
            if(agentIndex == state.getNumAgents()-1):
              tmpScore = max_value(nextState, alpha, beta, depth+1)[0]
            else:
              tmpScore = min_value(nextState, alpha, beta, agentIndex+1, depth)[0]
            nextScore = min(tmpScore, nextScore)
            if(tmpScore == nextScore):              
              nextAction = action            
            beta = min(beta, nextScore)
          return (nextScore, nextAction)

        return max_value(gameState, -float("inf"), float("inf"), 0)[1]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #assume pacman is maximizer
        #assume ghosts are expectimax
        def getExpectiMaxAction(gameState, agentIndex, depth):
          
          curScore = 0 #curScore will be avgScore for expectimax
          curAction = None          
          if(agentIndex == 0):
            curScore = -float("inf")

          actions = gameState.getLegalActions(agentIndex)
          if(gameState.isWin() or gameState.isLose() or len(actions) == 0 or (agentIndex == 0 and self.depth == depth)):
            return (self.evaluationFunction(gameState), None)
          for action in actions:
            nextState = gameState.generateSuccessor(agentIndex, action)
            if(agentIndex == gameState.getNumAgents()-1):
              nextScore = getExpectiMaxAction(nextState, 0, depth+1)[0] #maximizer
            else:
              nextScore = getExpectiMaxAction(nextState, agentIndex+1, depth)[0] #expectimaxizer
            if(agentIndex == 0):
              if(nextScore>curScore):
                curScore = nextScore
                curAction = action
            else:
              #if(nextScore<curScore):
                #curScore = nextScore
                curScore += nextScore/len(actions)
                #curAction = action
          return (curScore, curAction)

        return getExpectiMaxAction(gameState, 0, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I used the same evaluation function from ReflexAgent
      however, I updated to take into account scared ghosts
      as well, so that it would eat the scared ghosts
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
   
    if(successorGameState.isWin()):
      return float("inf");

    evalVal = 0    
    #evalVal = evalVal + 10*currentGameState.getNumFood()

    foodList = newFood.asList()
    nearestFood = util.manhattanDistance(foodList[0], newPos)
    for foodPos in foodList:
      foodDist = util.manhattanDistance(foodPos, newPos)
      if(foodDist<nearestFood):
        nearestFood = foodDist

    evalVal = evalVal-nearestFood #further the nearest food, lower the evaluation value      

    # distance to ghost, since max manhattan distance to ghost can be 2 before being eaten
    for ghost in newGhostStates:
      ghostPacDiff = util.manhattanDistance(ghostState.getPosition(), newPos)      
      if ghostState.scaredTimer > ghostPacDiff:
        evalVal += ghostState.scaredTimer + ghostPacDiff
      else:
        if(ghostPacDiff<3):
          evalVal = evalVal + ghostPacDiff
        else:
          evalVal = evalVal + 3

    evalVal = evalVal + successorGameState.getScore()

    return evalVal

# Abbreviation
better = betterEvaluationFunction

def evaluationFunction2(currentGameState):
  """
  Design a better evaluation function here.

  The evaluation function takes in the current and proposed successor
  GameStates (pacman.py) and returns a number, where higher numbers are better.

  The code below extracts some useful information from the state, like the
  remaining food (newFood) and Pacman position after moving (newPos).
  newScaredTimes holds the number of moves that each ghost will remain
  scared because of Pacman having eaten a power pellet.

  Print out these variables to see what you're getting, then combine them
  to create a masterful evaluation function.
  """
  # Useful information you can extract from a GameState (pacman.py)
  successorGameState = currentGameState
  newPos = successorGameState.getPacmanPosition()
  newFood = successorGameState.getFood()
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  "*** YOUR CODE HERE ***"
  if(successorGameState.isWin()):
    return float("inf");

  if(successorGameState.isLose()):
    return -float("inf");

  evalVal = float(0)

  foodList = newFood.asList()
  nearestFood = util.manhattanDistance(foodList[0], newPos)
  for foodPos in foodList:
    foodDist = util.manhattanDistance(foodPos, newPos)
    if(foodDist<nearestFood):
      nearestFood = foodDist

  evalVal = evalVal-nearestFood #further the nearest food, lower the evaluation value      

  # distance to ghost, since max manhattan distance to ghost can be 2
  for ghostState in newGhostStates:
    ghostPacDiff = util.manhattanDistance(ghostState.getPosition(), newPos)
    if ghostState.scaredTimer > ghostPacDiff:
      evalVal += 10*ghostState.scaredTimer - 10*ghostPacDiff
    else:
      if(ghostPacDiff<3):
        evalVal = evalVal + 10*ghostPacDiff
        if (currentGameState.getPacmanPosition() in currentGameState.getCapsules()):
          evalVal = evalVal + 50
      else:
        evalVal = evalVal + 30
        if (currentGameState.getPacmanPosition() in currentGameState.getCapsules()):
          evalVal = evalVal + 50

  if (currentGameState.getPacmanPosition() in foodList):
    evalVal = evalVal + 10

  

  evalVal = evalVal + 50*successorGameState.getScore()

  return evalVal

def evaluationFunction3(currentGameState, action):
  """
  Design a better evaluation function here.

  The evaluation function takes in the current and proposed successor
  GameStates (pacman.py) and returns a number, where higher numbers are better.

  The code below extracts some useful information from the state, like the
  remaining food (newFood) and Pacman position after moving (newPos).
  newScaredTimes holds the number of moves that each ghost will remain
  scared because of Pacman having eaten a power pellet.

  Print out these variables to see what you're getting, then combine them
  to create a masterful evaluation function.
  """
  # Useful information you can extract from a GameState (pacman.py)
  successorGameState = currentGameState.generatePacmanSuccessor(action)
  newPos = successorGameState.getPacmanPosition()
  newFood = successorGameState.getFood()
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  "*** YOUR CODE HERE ***"
  if(successorGameState.isWin()):
    return float("inf");

  evalVal = 0
  
  foodList = newFood.asList()
  nearestFood = util.manhattanDistance(foodList[0], newPos)
  for foodPos in foodList:
    foodDist = util.manhattanDistance(foodPos, newPos)
    if(foodDist<nearestFood):
      nearestFood = foodDist

  evalVal = evalVal-nearestFood #further the nearest food, lower the evaluation value      

  # distance to ghost, since max manhattan distance to ghost can be 2
  for x in range(1, currentGameState.getNumAgents() - 1):
    ghostPacDiff = util.manhattanDistance(currentGameState.getGhostPosition(x), newPos)
    if(ghostPacDiff<3):
      evalVal = evalVal + ghostPacDiff
    else:
      evalVal = evalVal + 3       
  
  if (currentGameState.getNumFood() > successorGameState.getNumFood()):
    evalVal = evalVal + 100

  evalVal = evalVal + successorGameState.getScore()

  return evalVal

#bool first = True
class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def getAction(self, gameState):
      """
        Returns an action.  You can use any method you want and search to any depth you want.
        Just remember that the mini-contest is timed, so you have to trade off speed and computation.

        Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
        just make a beeline straight towards Pacman (or away from him if they're scared!)
      """
      "*** YOUR CODE HERE ***"
      def max_value(state, alpha, beta, depth):          
        tmpScore = nextScore = -float("inf")
        nextAction = Directions.STOP
        actions = state.getLegalActions(0)
        if(state.isWin() or state.isLose() or len(actions) == 0 or self.depth == depth):
          return (evaluationFunction2(state), Directions.STOP)
        for action in actions:
          if (action != Directions.STOP):
            if(alpha>beta):
              return (nextScore, nextAction)
            nextState = state.generateSuccessor(0, action)
            tmpScore = min_value(nextState, alpha, beta, 1, depth)[0]
            nextScore = max(tmpScore, nextScore)
            if(tmpScore==nextScore):              
              nextAction = action            
            alpha = max(alpha, nextScore)
        return (nextScore, nextAction)

      def min_value(state, alpha, beta, agentIndex, depth):
        tmpScore = nextScore = float("inf")
        nextAction = Directions.STOP
        actions = state.getLegalActions(agentIndex)
        if(state.isWin() or state.isLose() or len(actions) == 0):
          return (evaluationFunction2(state), Directions.STOP)
        for action in actions:
          if(action != Directions.STOP):
            if(alpha>beta):
              return (nextScore, nextAction)
            nextState = state.generateSuccessor(agentIndex, action)
            if(agentIndex == state.getNumAgents()-1):
              tmpScore = max_value(nextState, alpha, beta, depth+1)[0]
            else:
              tmpScore = min_value(nextState, alpha, beta, agentIndex+1, depth)[0]
            nextScore = min(tmpScore, nextScore)
            if(tmpScore == nextScore):              
              nextAction = action            
            beta = min(beta, nextScore)
        return (nextScore, nextAction)

      def getExpectiMaxAction(gameState, agentIndex, depth):
          
        curScore = 0 #curScore will be avgScore for expectimax
        curAction = Directions.STOP          
        if(agentIndex == 0):
          curScore = -float("inf")

        actions = gameState.getLegalActions(agentIndex)
        if(gameState.isWin() or gameState.isLose() or len(actions) == 0 or (agentIndex == 0 and self.depth == depth)):
          return (evaluationFunction2(gameState), Directions.STOP)
        for action in actions:
          if(action != Directions.STOP):
            nextState = gameState.generateSuccessor(agentIndex, action)
            if(agentIndex == gameState.getNumAgents()-1):
              nextScore = getExpectiMaxAction(nextState, 0, depth+1)[0] #maximizer
            else:
              nextScore = getExpectiMaxAction(nextState, agentIndex+1, depth)[0] #expectimaxizer
            if(agentIndex == 0):
              if(nextScore>curScore):
                curScore = nextScore
                curAction = action
            else:
              #if(nextScore<curScore):
                #curScore = nextScore
                curScore += nextScore/len(actions)
                #curAction = action
        return (curScore, curAction)

      for ghostState in gameState.getGhostStates():
        ghostPacDiff = util.manhattanDistance(ghostState.getPosition(), gameState.getPacmanPosition())
        if ghostState.scaredTimer > ghostPacDiff:
          return getExpectiMaxAction(gameState, 0, 0)[1] 
        else:
          if(ghostPacDiff<3):
            return max_value(gameState, -float("inf"), float("inf"), 0)[1]
          else:
             return getExpectiMaxAction(gameState, 0, 0)[1]        
              
      
        

