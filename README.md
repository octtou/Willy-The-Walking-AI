## Intoruction
So its been a year, i havent made a new repo, ive been coding alot, while i still code on roblox i still do get better at coding. and on 23rd of Jully 2026 i made my first ever AI!!!! it was for a competition of making an essay and i got a bit of a scope creep. I couldn't belive myself that im able to make this with little to no AI involved, especialy for my first time and that theres not many materials on youtube about this kind of topic.

## The Project
My goal was to make a fully working AI that can learn to navigate a simple enviorment, literely just get to the end. I choose this because i know its my first time, so it doesnt need to navigate a whole maze

## How it Works
Willy (name of the AI) have 24 neurons, and 3 actions that he can make, he can go left, right, and forward, thinking about it now, why didnt i add 4 actions so he can go backward? maybe i was on something when making this, because again, its a miricle that im able to even make this.

```luau
NN:addLayer(8, false, "LeakyReLU")
NN:addLayer(16, true, "LeakyReLU")
NN:addLayer(3, false, "None")
```
Not going to lie, i did ask gemini on how this work, i basicaly asked gemini on how to make neural network by giving him the DataPredict document because english is my second language and theres ALOT of hard words i dont understand, it recomend me to use LeakyReLU, and to make the 16 neurons true, which i dont know why.

for the model itslef I used Deep-Q-Learning, because the document recomended me when i was browsing it for the first time, and Action list is the action list, so i just put {1, 2, 3} on it because theres 3 actions. actualy proud that this part i dint use AI, just discover it myself by reading the document.

```luau
local Model = DataPredict.Models.DeepQLearning.new({
	discountFactor = 0.99,
	EligibilityTrace = Eligibility
})

Model:setModel(NN)
Model:setActionsList({1, 2, 3})
NN:setClassesList({1,2,3})
```
not sure what the Classes list is, i added it because it doesnt work when i dont add it, and when i ask Gemini when debugging it, it said to use it.
also i noticed that theres an already built in ANN for DQL called DQN on the module, i dont think i read the document enough, should have use DQN not gonna lie.

This part of the code is for the reward system, i dont know why but the module doesnt seem to have the decay reward thing, gemini told me to add it when i was debugging the AI when the reward system broke. other than that epsilon, epsilonDecay, minEpsilon thing, the code for the reward system are mine. 

```luau
local sugar = 0

local epsilon = 1.0
local epsilonDecay = 0.995
local minEpsilon = 0.05

local Body = workspace.Willy
local Arm = 20 -- Studs!
local Root = Body:FindFirstChild("HumanoidRootPart")
local Humanoid = Body:FindFirstChild("Humanoid")

local Origin = Root.Position
local previousDistance = math.huge

local function getReward(rayState)
	local Goal = workspace.End
	local distance = (Root.Position - Goal.Position).Magnitude
	
	local delta = previousDistance - distance
	previousDistance = distance

	local reward = delta
	
	for i, WallDist in ipairs(rayState) do
		if WallDist < 0.08 then
			warn("IM SORRY!")
			sugar -= 10
			return -10, true -- gave trauma about walls
		end
	end
	
	if distance < 4 then
		warn("YUM!")
		sugar += 10
		return 50, true
	end
	
	print("Willies SUGAR: "..sugar)
	reward = math.clamp(reward, -1, 1)
	return reward - 0.05, false
end
```
Anyhow i decided to make sugar as the reward, I want willy to get addicted to sugar like its drugs to motivate him HARD. Willy basicaly have 8 raycast going around him that detects walls, and if he hit a wall he gets LESS sugar, also gemini recomend to me to add reward based on the distance between willy and the destination, so he gets sugar the closer he gets to the destination, wich i add, its the delta distance thingy.

Speaking of raycast, heres the part where i fully didnt use gemini. its the 8 way raycast, i actualy opened an old kinematic note from 11th grade to make this.

```luau
local function GetRaycast()
	local RayState = {}

	local RayP = RaycastParams.new()
	RayP.FilterType = Enum.RaycastFilterType.Exclude
	RayP.FilterDescendantsInstances = {Body, workspace.End}

	local directions = {
		Vector3.new(0,0,1).Unit,
		Vector3.new(1,0,1).Unit,
		Vector3.new(1,0,0).Unit,
		Vector3.new(1,0,-1).Unit,
		Vector3.new(0,0,-1).Unit,
		Vector3.new(-1,0,-1).Unit,
		Vector3.new(-1,0,0).Unit,
		Vector3.new(-1,0,1).Unit,
	}

	for index, offset in ipairs(directions) do

		local WorldRot = Root.CFrame:VectorToWorldSpace(offset) * Arm
		local Result = workspace:Raycast(Root.Position, WorldRot, RayP)

		local Debug = DebugBeams[index]

		if Result then
			table.insert(RayState, Result.Distance / Arm)

			Debug.End.WorldPosition = Result.Position
			Debug.Beam.Color = ColorSequence.new(Color3.fromRGB(255, 0, 0))

		else
			table.insert(RayState, 1.0)

			Debug.End.WorldPosition = Root.Position + WorldRot
			Debug.Beam.Color = ColorSequence.new(Color3.fromRGB(0, 255, 0))
		end
	end

	return RayState
end
```
Very proud of this code, work hard on it.

This is the heart of the code, this is the LOOP. basicaly it loops 1k times. the loop ends if Willy goes to the end, or hit a wall, or the episode reaches 1k.

```luau
task.spawn(function()
	local maxEpisodes = 1000
	
	for episode = 1, maxEpisodes do
		ResetWillyPosition()
		Model:reset()
		
		local currentState = GetRaycast()
		local isDone = false
		
		while not isDone do
			local bestAction

			if math.random() < epsilon then
				bestAction = math.random(1, 3)
			else
				local rawOutput = Model:predict({currentState})
				
				if typeof(rawOutput) == "table" then
					if typeof(rawOutput[1]) == "table" then
						bestAction = rawOutput[1][1]
					else
						bestAction = rawOutput[1]
					end
				elseif typeof(rawOutput) == "number" then
					bestAction = rawOutput
				end
				
				if not bestAction or typeof(bestAction) ~= "number" then
					warn("Predict returned:", bestAction)
					bestAction = math.random(1, 3)
				end
			end

			executeBotAction(bestAction)
			
			task.wait(0.15)
			
			local nextState = GetRaycast()
			local reward, done = getReward(nextState)
			warn("Reward:", reward, "Done:", done)
			isDone = done
			
			Model:categoricalUpdate(
				{currentState},      -- previous state
				bestAction,              -- previous action
				reward,              -- reward received
				{nextState},         -- current state
				nil,                 -- currentAction (unused by DQN)
				isDone and 1 or 0    -- terminalStateValue must be 0 or 1
			)
			
			currentState = nextState
		end
		
		epsilon = math.max(minEpsilon, epsilon * epsilonDecay)
		Model:episodeUpdate(isDone and 1 or 0)
		print("Episode: "..episode.." finished!")
	end
end)
```
Inside the loop theres many things, first I reset Willies position so hes at his starting point, and make sure isDone is false, otherwise the loop will be over. Willy then will make his 3 actions, eather go left, right, or straight. He pericts not randomly, but by checking his raycast, if he thinks theres no best action thats when hes going to choose randomly, his actions are recorded onto a table for next episodes learning, even after he reachses the end, he still will optimize his path himself. after that we gave him reward based on his performance, and then we get to the next Episode!

## Extra Stuff
https://github.com/user-attachments/assets/18b3aa00-af7b-4b6f-a247-62e9aba59c6d

WILLY MADE IT TO THE END IM SO PROUDD!!!!
