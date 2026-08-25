local SSS = game:GetService("ServerScriptService")
local DataPredict = require(SSS.DataPredict)

local NN = DataPredict.Models.NeuralNetwork.new({
	learningRate = 0.0005,
})

NN:addLayer(8, false, "LeakyReLU")
NN:addLayer(16, true, "LeakyReLU")
NN:addLayer(3, false, "None")

local Eligibility = DataPredict.EligibilityTraces.ReplacingTrace.new({
	lambda = 0.9,
})

local Model = DataPredict.Models.DeepQLearning.new({
	discountFactor = 0.99,
	EligibilityTrace = Eligibility
})

Model:setModel(NN)
Model:setActionsList({1, 2, 3})
NN:setClassesList({1,2,3})

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


local function ResetWillyPosition()
	warn("I go back")
	Body:PivotTo(CFrame.new(Origin))
	
	previousDistance = (Origin - workspace.End.Position).Magnitude
end

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

local function executeBotAction(action)
	if action == 1 then
		local targetPos = Root.Position + (Root.CFrame.LookVector * 4)
		Body:PivotTo(
			Root.CFrame + Root.CFrame.LookVector * 2
		)
	elseif action == 2 then
		warn("I go right.")
		Body:PivotTo(Root.CFrame * CFrame.Angles(0, math.rad(30), 0))
	elseif action == 3 then
		warn("I go left.")
		Body:PivotTo(Root.CFrame * CFrame.Angles(0, math.rad(-30), 0))
	end
end

local DebugBeams = {}

for i = 1, 8 do
	local a0 = Instance.new("Attachment")
	a0.Parent = Root

	local a1 = Instance.new("Attachment")
	a1.Parent = Root

	local beam = Instance.new("Beam")
	beam.Attachment0 = a0
	beam.Attachment1 = a1
	beam.Width0 = 0.08
	beam.Width1 = 0.08
	beam.FaceCamera = true
	beam.Parent = Root

	DebugBeams[i] = {
		Start = a0,
		End = a1,
		Beam = beam,
	}
end

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
