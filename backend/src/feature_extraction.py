"""
feature_extraction.py

Advanced feature extraction for Approximate Q-Learning agents.
These features help the agent learn better policies by capturing:
- Ghost threats (including scared ghosts that can be eaten)
- Food attraction and clustering
- Capsule/power pellet opportunities
- Escape routes and trapped situations
- Safe zones and dangerous areas

Used by ApproxQLearningAgent to compute state-action features for Q-value approximation.
"""

from util import Counter, manhattanDistance


def getAdvancedFeatures(state, action):
    """
    Extract comprehensive features from (state, action) pair.
    Returns a Counter of feature names -> values.
    
    Features are designed to be:
    1. Normalized (roughly 0-1 range)
    2. Sparse (many zeros for efficiency)
    3. Informative (capture key decision factors)
    """
    features = Counter()
    
    # Generate successor state
    successor = state.generatePacmanSuccessor(action)
    if successor is None:
        return features
    
    # Position after taking action
    pacPos = successor.getPacmanPosition()
    food = successor.getFood()
    foodList = food.asList()
    walls = successor.getWalls()
    ghostStates = successor.getGhostStates()
    capsules = successor.getCapsules()
    
    # ===============================
    # BIAS
    # ===============================
    features["bias"] = 1.0
    
    # ===============================
    # FOOD FEATURES
    # ===============================
    if foodList:
        # Closest food distance (normalized)
        minFoodDist = min(manhattanDistance(pacPos, f) for f in foodList)
        features["closestFood"] = 1.0 / (minFoodDist + 1)
        
        # Number of food pellets nearby (within 3 steps)
        nearbyFood = sum(1 for f in foodList if manhattanDistance(pacPos, f) <= 3)
        features["nearbyFood"] = nearbyFood / 10.0
        
        # Total food remaining (normalized)
        features["foodRemaining"] = len(foodList) / 50.0
        
        # Moving towards food?
        currentPos = state.getPacmanPosition()
        currentMinDist = min(manhattanDistance(currentPos, f) for f in foodList)
        if minFoodDist < currentMinDist:
            features["movingToFood"] = 1.0
    
    # ===============================
    # GHOST FEATURES
    # ===============================
    if ghostStates:
        # Separate scared and active ghosts
        scaredGhosts = [g for g in ghostStates if g.scaredTimer > 0]
        activeGhosts = [g for g in ghostStates if g.scaredTimer <= 0]
        
        # Active ghost threats
        if activeGhosts:
            activePositions = [g.getPosition() for g in activeGhosts]
            activeDists = [manhattanDistance(pacPos, pos) for pos in activePositions]
            minActiveDist = min(activeDists)
            
            # Inverse distance to closest active ghost
            features["closestActiveGhost"] = 1.0 / (minActiveDist + 1)
            
            # Danger zone (ghost within 2 steps)
            if minActiveDist <= 2:
                features["danger"] = 1.0
            if minActiveDist <= 1:
                features["immediateDanger"] = 1.0
            
            # Number of nearby active ghosts
            nearbyActive = sum(1 for d in activeDists if d <= 3)
            features["nearbyActiveGhosts"] = nearbyActive / 4.0
            
            # Ghost direction (is ghost moving towards us?)
            # This helps predict future ghost positions
            ghostDirs = [g.getDirection() for g in activeGhosts]
            features["ghostsApproaching"] = sum(1 for d in ghostDirs if d != 'Stop') / 4.0
        
        # Scared ghosts (can be eaten for bonus)
        if scaredGhosts:
            scaredPositions = [g.getPosition() for g in scaredGhosts]
            scaredDists = [manhattanDistance(pacPos, pos) for pos in scaredPositions]
            minScaredDist = min(scaredDists)
            
            # Distance to closest scared ghost
            features["closestScaredGhost"] = 1.0 / (minScaredDist + 1)
            
            # Can we reach scared ghost before timer runs out?
            minScaredTimer = min(g.scaredTimer for g in scaredGhosts)
            if minScaredDist <= minScaredTimer:
                features["canEatGhost"] = 1.0
            
            # Moving towards scared ghost?
            currentPos = state.getPacmanPosition()
            currentScaredDists = [manhattanDistance(currentPos, pos) for pos in scaredPositions]
            currentMinScared = min(currentScaredDists)
            if minScaredDist < currentMinScared:
                features["chasingScaredGhost"] = 1.0
    
    # ===============================
    # CAPSULE FEATURES
    # ===============================
    if capsules:
        capsuleDists = [manhattanDistance(pacPos, c) for c in capsules]
        minCapsuleDist = min(capsuleDists)
        
        # Distance to closest capsule
        features["closestCapsule"] = 1.0 / (minCapsuleDist + 1)
        
        # Is capsule worth going for? (active ghosts nearby)
        if ghostStates:
            activeGhosts = [g for g in ghostStates if g.scaredTimer <= 0]
            if activeGhosts and minCapsuleDist <= 5:
                features["capsuleOpportunity"] = 1.0
        
        # Moving towards capsule?
        currentPos = state.getPacmanPosition()
        currentCapsuleDists = [manhattanDistance(currentPos, c) for c in capsules]
        if minCapsuleDist < min(currentCapsuleDists):
            features["movingToCapsule"] = 1.0
    
    # ===============================
    # ESCAPE/SAFETY FEATURES
    # ===============================
    legalActions = successor.getLegalPacmanActions()
    
    # Number of escape routes (legal moves)
    features["numEscapeRoutes"] = len(legalActions) / 5.0
    
    # Are we in a tunnel/corridor? (limited escape routes)
    if len(legalActions) <= 2:
        features["inTunnel"] = 1.0
    
    # Dead end detection
    if len(legalActions) == 1:
        features["deadEnd"] = 1.0
    
    # Stop action penalty
    if action == 'Stop':
        features["stopped"] = 1.0
    
    # ===============================
    # SCORE FEATURES
    # ===============================
    features["score"] = successor.getScore() / 1000.0
    
    return features


def getSimpleFeatures(state, action):
    """
    Simpler feature set for faster learning.
    Good for small maps and quick training.
    """
    features = Counter()
    
    successor = state.generatePacmanSuccessor(action)
    if successor is None:
        return features
    
    pacPos = successor.getPacmanPosition()
    foodList = successor.getFood().asList()
    ghostStates = successor.getGhostStates()
    capsules = successor.getCapsules()
    
    features["bias"] = 1.0
    
    # Food
    if foodList:
        minFoodDist = min(manhattanDistance(pacPos, f) for f in foodList)
        features["closestFood"] = 1.0 / (minFoodDist + 1)
    
    # Ghosts
    activeGhosts = [g for g in ghostStates if g.scaredTimer <= 0]
    scaredGhosts = [g for g in ghostStates if g.scaredTimer > 0]
    
    if activeGhosts:
        activeDists = [manhattanDistance(pacPos, g.getPosition()) for g in activeGhosts]
        minActive = min(activeDists)
        features["closestGhost"] = 1.0 / (minActive + 1)
        if minActive <= 1:
            features["danger"] = 1.0
    
    if scaredGhosts:
        scaredDists = [manhattanDistance(pacPos, g.getPosition()) for g in scaredGhosts]
        minScared = min(scaredDists)
        features["closestScared"] = 1.0 / (minScared + 1)
        if minScared <= 2:
            features["eatGhost"] = 1.0
    
    # Capsules
    if capsules:
        minCapDist = min(manhattanDistance(pacPos, c) for c in capsules)
        features["closestCapsule"] = 1.0 / (minCapDist + 1)
    
    # Stop penalty
    if action == 'Stop':
        features["stopped"] = 1.0
    
    return features


def getCompetitionFeatures(state, action):
    """
    Competition-grade features optimized for high win rates.
    Combines best elements from advanced and simple feature sets.
    """
    features = Counter()
    
    successor = state.generatePacmanSuccessor(action)
    if successor is None:
        return features
    
    pacPos = successor.getPacmanPosition()
    foodList = successor.getFood().asList()
    ghostStates = successor.getGhostStates()
    capsules = successor.getCapsules()
    legalActions = successor.getLegalPacmanActions()

    prevFoodCount = state.getNumFood()
    newFoodCount = successor.getNumFood()
    prevCapsules = state.getCapsules()
    prevCapsuleCount = len(prevCapsules) if prevCapsules is not None else 0
    newCapsuleCount = len(capsules) if capsules is not None else 0
    
    # Bias
    features["bias"] = 1.0

    if newFoodCount < prevFoodCount:
        features["ateFood"] = 1.0

    if newCapsuleCount < prevCapsuleCount:
        features["ateCapsule"] = 1.0
    
    # === FOOD ===
    if foodList:
        minFoodDist = min(manhattanDistance(pacPos, f) for f in foodList)
        features["closestFood"] = 1.0 / (minFoodDist + 1)
        
        # Food density nearby
        nearbyFood = sum(1 for f in foodList if manhattanDistance(pacPos, f) <= 3)
        features["foodDensity"] = nearbyFood / 8.0
    
    # === GHOSTS ===
    activeGhosts = [g for g in ghostStates if g.scaredTimer <= 0]
    scaredGhosts = [g for g in ghostStates if g.scaredTimer > 0]
    
    if activeGhosts:
        activeDists = [manhattanDistance(pacPos, g.getPosition()) for g in activeGhosts]
        minActive = min(activeDists)
        
        # Critical danger features
        features["ghostDist"] = 1.0 / (minActive + 1)
        
        if minActive <= 2:
            features["nearGhost"] = 1.0
        if minActive <= 1:
            features["danger"] = 1.0
        
        # Multiple ghost threat
        closeGhosts = sum(1 for d in activeDists if d <= 3)
        features["ghostPressure"] = closeGhosts / 4.0

        if minActive <= 4:
            features["avoidGhost"] = 1.0 / (minActive + 1)
    
    if scaredGhosts:
        scaredDists = [manhattanDistance(pacPos, g.getPosition()) for g in scaredGhosts]
        minScared = min(scaredDists)
        minTimer = min(g.scaredTimer for g in scaredGhosts)
        
        features["scaredGhostDist"] = 1.0 / (minScared + 1)
        
        # Can reach before timer expires?
        if minScared < minTimer:
            features["canEatGhost"] = 1.0
    
    # === CAPSULES ===
    if capsules:
        minCapDist = min(manhattanDistance(pacPos, c) for c in capsules)
        features["capsuleDist"] = 1.0 / (minCapDist + 1)
        
        # Strategic capsule collection
        if activeGhosts and minCapDist <= 4:
            features["strategicCapsule"] = 1.0
    
    # === MOBILITY ===
    features["mobility"] = len(legalActions) / 5.0
    
    if len(legalActions) <= 2:
        features["restricted"] = 1.0
    
    # === ACTION PENALTIES ===
    if action == 'Stop':
        features["stopped"] = 1.0

    rev = {"North": "South", "South": "North", "East": "West", "West": "East"}
    prevDir = state.getPacmanState().configuration.direction
    if action in rev and prevDir in rev and action == rev[prevDir]:
        features["reverse"] = 1.0
    
    # === GAME STATE ===
    features["foodRemaining"] = len(foodList) / 50.0
    features["score"] = successor.getScore() / 500.0
    
    return features
