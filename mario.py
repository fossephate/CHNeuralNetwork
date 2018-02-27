# MarI/O by SethBling
# Feel free to use this code, but please do not redistribute it.
# Inted for use with the BizHawk emulator and Super Mario World or Super Mario Bros. ROM.
# For SMW, make sure you have a save state named "DP1.state" at the beginning of a level,
# and put a copy in both the Lua folder and the root directory of BizHawk.


Filename = "CloneHero-1.state"
ButtonNames = [
	"A",
	"S",
	"J",
	"K",
	"L",
	"Up",
	"Down",
]


BoxRadius = 6
InputSize = (BoxRadius*2+1)*(BoxRadius*2+1)

Inputs = InputSize+1
Outputs = None#ButtonNames

Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

TimeoutConstant = 20

MaxNodes = 1000000

def getPositions():
	if gameinfo.getromname() == "Super Mario World (USA)":
		marioX = memory.read_s16_le(0x94)
		marioY = memory.read_s16_le(0x96)
		
		local layer1x = memory.read_s16_le(0x1A);
		local layer1y = memory.read_s16_le(0x1C);
		
		screenX = marioX-layer1x
		screenY = marioY-layer1y
	elseif gameinfo.getromname() == "Super Mario Bros.":
		marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86)
		marioY = memory.readbyte(0x03B8)+16
	
		screenX = memory.readbyte(0x03AD)
		screenY = memory.readbyte(0x03B8)
	


def getTile(dx, dy):
	if gameinfo.getromname() == "Super Mario World (USA)":
		x = math.floor((marioX+dx+8)/16)
		y = math.floor((marioY+dy)/16)
		
		return memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10)
	elseif gameinfo.getromname() == "Super Mario Bros.":
		local x = marioX + dx + 8
		local y = marioY + dy - 16
		local page = math.floor(x/256)%2

		local subx = math.floor((x%256)/16)
		local suby = math.floor((y - 32)/16)
		local addr = 0x500 + page*13*16+suby*16+subx
		
		if suby >= 13 or suby < 0:
			return 0
		
		
		if memory.readbyte(addr) ~= 0:
			return 1
		else
			return 0
		
	


def getSprites():
	if gameinfo.getromname() == "Super Mario World (USA)":
		local sprites = {}
		for slot=0,11 do
			local status = memory.readbyte(0x14C8+slot)
			if status ~= 0:
				spritex = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256
				spritey = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256
				sprites[#sprites+1] = {["x"]=spritex, ["y"]=spritey}
			
				
		
		return sprites
	elseif gameinfo.getromname() == "Super Mario Bros.":
		local sprites = {}
		for slot=0,4 do
			local enemy = memory.readbyte(0xF+slot)
			if enemy ~= 0:
				local ex = memory.readbyte(0x6E + slot)*0x100 + memory.readbyte(0x87+slot)
				local ey = memory.readbyte(0xCF + slot)+24
				sprites[#sprites+1] = {["x"]=ex,["y"]=ey}
			
		
		
		return sprites
	


def getExtedSprites():
	if gameinfo.getromname() == "Super Mario World (USA)":
		local exted = {}
		for slot=0,11 do
			local number = memory.readbyte(0x170B+slot)
			if number ~= 0:
				spritex = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256
				spritey = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256
				exted[#exted+1] = {["x"]=spritex, ["y"]=spritey}
			
				
		
		return exted
	elseif gameinfo.getromname() == "Super Mario Bros.":
		return {}
	


def getInputs():
	getPositions()
	
	sprites = getSprites()
	exted = getExtedSprites()
	
	local inputs = {}
	
	for dy=-BoxRadius*16,BoxRadius*16,16 do
		for dx=-BoxRadius*16,BoxRadius*16,16 do
			inputs[#inputs+1] = 0
			
			tile = getTile(dx, dy)
			if tile == 1 and marioY+dy < 0x1B0:
				inputs[#inputs] = 1
			
			
			for i = 1,#sprites do
				distx = math.abs(sprites[i]["x"] - (marioX+dx))
				disty = math.abs(sprites[i]["y"] - (marioY+dy))
				if distx <= 8 and disty <= 8:
					inputs[#inputs] = -1
				
			

			for i = 1,#exted do
				distx = math.abs(exted[i]["x"] - (marioX+dx))
				disty = math.abs(exted[i]["y"] - (marioY+dy))
				if distx < 8 and disty < 8:
					inputs[#inputs] = -1
				
			
		
	
	
	#mariovx = memory.read_s8(0x7B)
	#mariovy = memory.read_s8(0x7D)
	
	return inputs


def sigmoid(x):
	return 2/(1+math.exp(-4.9*x))-1


def newInnovation():
	pool.innovation = pool.innovation + 1
	return pool.innovation


def newPool():
	local pool = {}
	pool.species = {}
	pool.generation = 0
	pool.innovation = Outputs
	pool.currentSpecies = 1
	pool.currentGenome = 1
	pool.currentFrame = 0
	pool.maxFitness = 0
	
	return pool


def newSpecies():
	local species = {}
	species.topFitness = 0
	species.staleness = 0
	species.genomes = {}
	species.averageFitness = 0
	
	return species


def newGenome():
	local genome = {}
	genome.genes = {}
	genome.fitness = 0
	genome.adjustedFitness = 0
	genome.network = {}
	genome.maxneuron = 0
	genome.globalRank = 0
	genome.mutationRates = {}
	genome.mutationRates["connections"] = MutateConnectionsChance
	genome.mutationRates["link"] = LinkMutationChance
	genome.mutationRates["bias"] = BiasMutationChance
	genome.mutationRates["node"] = NodeMutationChance
	genome.mutationRates["enable"] = EnableMutationChance
	genome.mutationRates["disable"] = DisableMutationChance
	genome.mutationRates["step"] = StepSize
	
	return genome


def copyGenome(genome)
	local genome2 = newGenome()
	for g=1,#genome.genes do
		table.insert(genome2.genes, copyGene(genome.genes[g]))
	
	genome2.maxneuron = genome.maxneuron
	genome2.mutationRates["connections"] = genome.mutationRates["connections"]
	genome2.mutationRates["link"] = genome.mutationRates["link"]
	genome2.mutationRates["bias"] = genome.mutationRates["bias"]
	genome2.mutationRates["node"] = genome.mutationRates["node"]
	genome2.mutationRates["enable"] = genome.mutationRates["enable"]
	genome2.mutationRates["disable"] = genome.mutationRates["disable"]
	
	return genome2


def basicGenome()
	local genome = newGenome()
	local innovation = 1

	genome.maxneuron = Inputs
	mutate(genome)
	
	return genome


def newGene()
	local gene = {}
	gene.into = 0
	gene.out = 0
	gene.weight = 0.0
	gene.enabled = true
	gene.innovation = 0
	
	return gene


def copyGene(gene)
	local gene2 = newGene()
	gene2.into = gene.into
	gene2.out = gene.out
	gene2.weight = gene.weight
	gene2.enabled = gene.enabled
	gene2.innovation = gene.innovation
	
	return gene2


def newNeuron()
	local neuron = {}
	neuron.incoming = {}
	neuron.value = 0.0
	
	return neuron


def generateNetwork(genome)
	local network = {}
	network.neurons = {}
	
	for i=1,Inputs do
		network.neurons[i] = newNeuron()
	
	
	for o=1,Outputs do
		network.neurons[MaxNodes+o] = newNeuron()
	
	
	table.sort(genome.genes, def (a,b)
		return (a.out < b.out)
	)
	for i=1,#genome.genes do
		local gene = genome.genes[i]
		if gene.enabled:
			if network.neurons[gene.out] == nil:
				network.neurons[gene.out] = newNeuron()
			
			local neuron = network.neurons[gene.out]
			table.insert(neuron.incoming, gene)
			if network.neurons[gene.into] == nil:
				network.neurons[gene.into] = newNeuron()
			
		
	
	
	genome.network = network


def evaluateNetwork(network, inputs)
	table.insert(inputs, 1)
	if #inputs ~= Inputs:
		console.writeline("Incorrect number of neural network inputs.")
		return {}
	
	
	for i=1,Inputs do
		network.neurons[i].value = inputs[i]
	
	
	for _,neuron in pairs(network.neurons) do
		local sum = 0
		for j = 1,#neuron.incoming do
			local incoming = neuron.incoming[j]
			local other = network.neurons[incoming.into]
			sum = sum + incoming.weight * other.value
		
		
		if #neuron.incoming > 0:
			neuron.value = sigmoid(sum)
		
	
	
	local outputs = {}
	for o=1,Outputs do
		local button = "P1 " .. ButtonNames[o]
		if network.neurons[MaxNodes+o].value > 0:
			outputs[button] = true
		else
			outputs[button] = false
		
	
	
	return outputs


def crossover(g1, g2)
	# Make sure g1 is the higher fitness genome
	if g2.fitness > g1.fitness:
		tempg = g1
		g1 = g2
		g2 = tempg
	

	local child = newGenome()
	
	local innovations2 = {}
	for i=1,#g2.genes do
		local gene = g2.genes[i]
		innovations2[gene.innovation] = gene
	
	
	for i=1,#g1.genes do
		local gene1 = g1.genes[i]
		local gene2 = innovations2[gene1.innovation]
		if gene2 ~= nil and math.random(2) == 1 and gene2.enabled:
			table.insert(child.genes, copyGene(gene2))
		else
			table.insert(child.genes, copyGene(gene1))
		
	
	
	child.maxneuron = math.max(g1.maxneuron,g2.maxneuron)
	
	for mutation,rate in pairs(g1.mutationRates) do
		child.mutationRates[mutation] = rate
	
	
	return child


def randomNeuron(genes, nonInput)
	local neurons = {}
	if not nonInput:
		for i=1,Inputs do
			neurons[i] = true
		
	
	for o=1,Outputs do
		neurons[MaxNodes+o] = true
	
	for i=1,#genes do
		if (not nonInput) or genes[i].into > Inputs:
			neurons[genes[i].into] = true
		
		if (not nonInput) or genes[i].out > Inputs:
			neurons[genes[i].out] = true
		
	

	local count = 0
	for _,_ in pairs(neurons) do
		count = count + 1
	
	local n = math.random(1, count)
	
	for k,v in pairs(neurons) do
		n = n-1
		if n == 0:
			return k
		
	
	
	return 0


def containsLink(genes, link)
	for i=1,#genes do
		local gene = genes[i]
		if gene.into == link.into and gene.out == link.out:
			return true
		
	


def pointMutate(genome)
	local step = genome.mutationRates["step"]
	
	for i=1,#genome.genes do
		local gene = genome.genes[i]
		if math.random() < PerturbChance:
			gene.weight = gene.weight + math.random() * step*2 - step
		else
			gene.weight = math.random()*4-2
		
	


def linkMutate(genome, forceBias)
	local neuron1 = randomNeuron(genome.genes, false)
	local neuron2 = randomNeuron(genome.genes, true)
	 
	local newLink = newGene()
	if neuron1 <= Inputs and neuron2 <= Inputs:
		#Both input nodes
		return
	
	if neuron2 <= Inputs:
		# Swap output and input
		local temp = neuron1
		neuron1 = neuron2
		neuron2 = temp
	

	newLink.into = neuron1
	newLink.out = neuron2
	if forceBias:
		newLink.into = Inputs
	
	
	if containsLink(genome.genes, newLink):
		return
	
	newLink.innovation = newInnovation()
	newLink.weight = math.random()*4-2
	
	table.insert(genome.genes, newLink)


def nodeMutate(genome)
	if #genome.genes == 0:
		return
	

	genome.maxneuron = genome.maxneuron + 1

	local gene = genome.genes[math.random(1,#genome.genes)]
	if not gene.enabled:
		return
	
	gene.enabled = false
	
	local gene1 = copyGene(gene)
	gene1.out = genome.maxneuron
	gene1.weight = 1.0
	gene1.innovation = newInnovation()
	gene1.enabled = true
	table.insert(genome.genes, gene1)
	
	local gene2 = copyGene(gene)
	gene2.into = genome.maxneuron
	gene2.innovation = newInnovation()
	gene2.enabled = true
	table.insert(genome.genes, gene2)


def enableDisableMutate(genome, enable)
	local candidates = {}
	for _,gene in pairs(genome.genes) do
		if gene.enabled == not enable:
			table.insert(candidates, gene)
		
	
	
	if #candidates == 0:
		return
	
	
	local gene = candidates[math.random(1,#candidates)]
	gene.enabled = not gene.enabled


def mutate(genome)
	for mutation,rate in pairs(genome.mutationRates) do
		if math.random(1,2) == 1:
			genome.mutationRates[mutation] = 0.95*rate
		else
			genome.mutationRates[mutation] = 1.05263*rate
		
	

	if math.random() < genome.mutationRates["connections"]:
		pointMutate(genome)
	
	
	local p = genome.mutationRates["link"]
	while p > 0 do
		if math.random() < p:
			linkMutate(genome, false)
		
		p = p - 1
	

	p = genome.mutationRates["bias"]
	while p > 0 do
		if math.random() < p:
			linkMutate(genome, true)
		
		p = p - 1
	
	
	p = genome.mutationRates["node"]
	while p > 0 do
		if math.random() < p:
			nodeMutate(genome)
		
		p = p - 1
	
	
	p = genome.mutationRates["enable"]
	while p > 0 do
		if math.random() < p:
			enableDisableMutate(genome, true)
		
		p = p - 1
	

	p = genome.mutationRates["disable"]
	while p > 0 do
		if math.random() < p:
			enableDisableMutate(genome, false)
		
		p = p - 1
	


def disjoint(genes1, genes2)
	local i1 = {}
	for i = 1,#genes1 do
		local gene = genes1[i]
		i1[gene.innovation] = true
	

	local i2 = {}
	for i = 1,#genes2 do
		local gene = genes2[i]
		i2[gene.innovation] = true
	
	
	local disjointGenes = 0
	for i = 1,#genes1 do
		local gene = genes1[i]
		if not i2[gene.innovation]:
			disjointGenes = disjointGenes+1
		
	
	
	for i = 1,#genes2 do
		local gene = genes2[i]
		if not i1[gene.innovation]:
			disjointGenes = disjointGenes+1
		
	
	
	local n = math.max(#genes1, #genes2)
	
	return disjointGenes / n


def weights(genes1, genes2)
	local i2 = {}
	for i = 1,#genes2 do
		local gene = genes2[i]
		i2[gene.innovation] = gene
	

	local sum = 0
	local coincident = 0
	for i = 1,#genes1 do
		local gene = genes1[i]
		if i2[gene.innovation] ~= nil:
			local gene2 = i2[gene.innovation]
			sum = sum + math.abs(gene.weight - gene2.weight)
			coincident = coincident + 1
		
	
	
	return sum / coincident

	
def sameSpecies(genome1, genome2)
	local dd = DeltaDisjoint*disjoint(genome1.genes, genome2.genes)
	local dw = DeltaWeights*weights(genome1.genes, genome2.genes) 
	return dd + dw < DeltaThreshold


def rankGlobally()
	local global = {}
	for s = 1,#pool.species do
		local species = pool.species[s]
		for g = 1,#species.genomes do
			table.insert(global, species.genomes[g])
		
	
	table.sort(global, def (a,b)
		return (a.fitness < b.fitness)
	)
	
	for g=1,#global do
		global[g].globalRank = g
	


def calculateAverageFitness(species)
	local total = 0
	
	for g=1,#species.genomes do
		local genome = species.genomes[g]
		total = total + genome.globalRank
	
	
	species.averageFitness = total / #species.genomes


def totalAverageFitness()
	local total = 0
	for s = 1,#pool.species do
		local species = pool.species[s]
		total = total + species.averageFitness
	

	return total


def cullSpecies(cutToOne)
	for s = 1,#pool.species do
		local species = pool.species[s]
		
		table.sort(species.genomes, def (a,b)
			return (a.fitness > b.fitness)
		)
		
		local remaining = math.ceil(#species.genomes/2)
		if cutToOne:
			remaining = 1
		
		while #species.genomes > remaining do
			table.remove(species.genomes)
		
	


def breedChild(species)
	local child = {}
	if math.random() < CrossoverChance:
		g1 = species.genomes[math.random(1, #species.genomes)]
		g2 = species.genomes[math.random(1, #species.genomes)]
		child = crossover(g1, g2)
	else
		g = species.genomes[math.random(1, #species.genomes)]
		child = copyGenome(g)
	
	
	mutate(child)
	
	return child


def removeStaleSpecies()
	local survived = {}

	for s = 1,#pool.species do
		local species = pool.species[s]
		
		table.sort(species.genomes, def (a,b)
			return (a.fitness > b.fitness)
		)
		
		if species.genomes[1].fitness > species.topFitness:
			species.topFitness = species.genomes[1].fitness
			species.staleness = 0
		else
			species.staleness = species.staleness + 1
		
		if species.staleness < StaleSpecies or species.topFitness >= pool.maxFitness:
			table.insert(survived, species)
		
	

	pool.species = survived


def removeWeakSpecies()
	local survived = {}

	local sum = totalAverageFitness()
	for s = 1,#pool.species do
		local species = pool.species[s]
		breed = math.floor(species.averageFitness / sum * Population)
		if breed >= 1:
			table.insert(survived, species)
		
	

	pool.species = survived



def addToSpecies(child)
	local foundSpecies = false
	for s=1,#pool.species do
		local species = pool.species[s]
		if not foundSpecies and sameSpecies(child, species.genomes[1]):
			table.insert(species.genomes, child)
			foundSpecies = true
		
	
	
	if not foundSpecies:
		local childSpecies = newSpecies()
		table.insert(childSpecies.genomes, child)
		table.insert(pool.species, childSpecies)
	


def newGeneration()
	cullSpecies(false) # Cull the bottom half of each species
	rankGlobally()
	removeStaleSpecies()
	rankGlobally()
	for s = 1,#pool.species do
		local species = pool.species[s]
		calculateAverageFitness(species)
	
	removeWeakSpecies()
	local sum = totalAverageFitness()
	local children = {}
	for s = 1,#pool.species do
		local species = pool.species[s]
		breed = math.floor(species.averageFitness / sum * Population) - 1
		for i=1,breed do
			table.insert(children, breedChild(species))
		
	
	cullSpecies(true) # Cull all but the top member of each species
	while #children + #pool.species < Population do
		local species = pool.species[math.random(1, #pool.species)]
		table.insert(children, breedChild(species))
	
	for c=1,#children do
		local child = children[c]
		addToSpecies(child)
	
	
	pool.generation = pool.generation + 1
	
	writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile))

	
def initializePool()
	pool = newPool()

	for i=1,Population do
		basic = basicGenome()
		addToSpecies(basic)
	

	initializeRun()


def clearJoypad()
	controller = {}
	for b = 1,#ButtonNames do
		controller["P1 " .. ButtonNames[b]] = false
	
	joypad.set(controller)


def initializeRun()
	savestate.load(Filename);
	rightmost = 0
	pool.currentFrame = 0
	timeout = TimeoutConstant
	clearJoypad()
	
	local species = pool.species[pool.currentSpecies]
	local genome = species.genomes[pool.currentGenome]
	generateNetwork(genome)
	evaluateCurrent()


def evaluateCurrent()
	local species = pool.species[pool.currentSpecies]
	local genome = species.genomes[pool.currentGenome]

	inputs = getInputs()
	controller = evaluateNetwork(genome.network, inputs)
	
	if controller["P1 Left"] and controller["P1 Right"]:
		controller["P1 Left"] = false
		controller["P1 Right"] = false
	
	if controller["P1 Up"] and controller["P1 Down"]:
		controller["P1 Up"] = false
		controller["P1 Down"] = false
	

	joypad.set(controller)


if pool == nil:
	initializePool()



def nextGenome()
	pool.currentGenome = pool.currentGenome + 1
	if pool.currentGenome > #pool.species[pool.currentSpecies].genomes:
		pool.currentGenome = 1
		pool.currentSpecies = pool.currentSpecies+1
		if pool.currentSpecies > #pool.species:
			newGeneration()
			pool.currentSpecies = 1
		
	


def fitnessAlreadyMeasured()
	local species = pool.species[pool.currentSpecies]
	local genome = species.genomes[pool.currentGenome]
	
	return genome.fitness ~= 0


def displayGenome(genome)
	local network = genome.network
	local cells = {}
	local i = 1
	local cell = {}
	for dy=-BoxRadius,BoxRadius do
		for dx=-BoxRadius,BoxRadius do
			cell = {}
			cell.x = 50+5*dx
			cell.y = 70+5*dy
			cell.value = network.neurons[i].value
			cells[i] = cell
			i = i + 1
		
	
	local biasCell = {}
	biasCell.x = 80
	biasCell.y = 110
	biasCell.value = network.neurons[Inputs].value
	cells[Inputs] = biasCell
	
	for o = 1,Outputs do
		cell = {}
		cell.x = 220
		cell.y = 30 + 8 * o
		cell.value = network.neurons[MaxNodes + o].value
		cells[MaxNodes+o] = cell
		local color
		if cell.value > 0:
			color = 0xFF0000FF
		else
			color = 0xFF000000
		
		gui.drawText(223, 24+8*o, ButtonNames[o], color, 9)
	
	
	for n,neuron in pairs(network.neurons) do
		cell = {}
		if n > Inputs and n <= MaxNodes:
			cell.x = 140
			cell.y = 40
			cell.value = neuron.value
			cells[n] = cell
		
	
	
	for n=1,4 do
		for _,gene in pairs(genome.genes) do
			if gene.enabled:
				local c1 = cells[gene.into]
				local c2 = cells[gene.out]
				if gene.into > Inputs and gene.into <= MaxNodes:
					c1.x = 0.75*c1.x + 0.25*c2.x
					if c1.x >= c2.x:
						c1.x = c1.x - 40
					
					if c1.x < 90:
						c1.x = 90
					
					
					if c1.x > 220:
						c1.x = 220
					
					c1.y = 0.75*c1.y + 0.25*c2.y
					
				
				if gene.out > Inputs and gene.out <= MaxNodes:
					c2.x = 0.25*c1.x + 0.75*c2.x
					if c1.x >= c2.x:
						c2.x = c2.x + 40
					
					if c2.x < 90:
						c2.x = 90
					
					if c2.x > 220:
						c2.x = 220
					
					c2.y = 0.25*c1.y + 0.75*c2.y
				
			
		
	
	
	gui.drawBox(50-BoxRadius*5-3,70-BoxRadius*5-3,50+BoxRadius*5+2,70+BoxRadius*5+2,0xFF000000, 0x80808080)
	for n,cell in pairs(cells) do
		if n > Inputs or cell.value ~= 0:
			local color = math.floor((cell.value+1)/2*256)
			if color > 255: color = 255 
			if color < 0: color = 0 
			local opacity = 0xFF000000
			if cell.value == 0:
				opacity = 0x50000000
			
			color = opacity + color*0x10000 + color*0x100 + color
			gui.drawBox(cell.x-2,cell.y-2,cell.x+2,cell.y+2,opacity,color)
		
	
	for _,gene in pairs(genome.genes) do
		if gene.enabled:
			local c1 = cells[gene.into]
			local c2 = cells[gene.out]
			local opacity = 0xA0000000
			if c1.value == 0:
				opacity = 0x20000000
			
			
			local color = 0x80-math.floor(math.abs(sigmoid(gene.weight))*0x80)
			if gene.weight > 0: 
				color = opacity + 0x8000 + 0x10000*color
			else
				color = opacity + 0x800000 + 0x100*color
			
			gui.drawLine(c1.x+1, c1.y, c2.x-3, c2.y, color)
		
	
	
	gui.drawBox(49,71,51,78,0x00000000,0x80FF0000)
	
	if forms.ischecked(showMutationRates):
		local pos = 100
		for mutation,rate in pairs(genome.mutationRates) do
			gui.drawText(100, pos, mutation .. ": " .. rate, 0xFF000000, 10)
			pos = pos + 8
		
	


def writeFile(filename)
        local file = io.open(filename, "w")
	file:write(pool.generation .. "\n")
	file:write(pool.maxFitness .. "\n")
	file:write(#pool.species .. "\n")
        for n,species in pairs(pool.species) do
		file:write(species.topFitness .. "\n")
		file:write(species.staleness .. "\n")
		file:write(#species.genomes .. "\n")
		for m,genome in pairs(species.genomes) do
			file:write(genome.fitness .. "\n")
			file:write(genome.maxneuron .. "\n")
			for mutation,rate in pairs(genome.mutationRates) do
				file:write(mutation .. "\n")
				file:write(rate .. "\n")
			
			file:write("done\n")
			
			file:write(#genome.genes .. "\n")
			for l,gene in pairs(genome.genes) do
				file:write(gene.into .. " ")
				file:write(gene.out .. " ")
				file:write(gene.weight .. " ")
				file:write(gene.innovation .. " ")
				if(gene.enabled):
					file:write("1\n")
				else
					file:write("0\n")
				
			
		
        
        file:close()


def savePool()
	local filename = forms.gettext(saveLoadFile)
	writeFile(filename)


def loadFile(filename)
        local file = io.open(filename, "r")
	pool = newPool()
	pool.generation = file:read("*number")
	pool.maxFitness = file:read("*number")
	forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
        local numSpecies = file:read("*number")
        for s=1,numSpecies do
		local species = newSpecies()
		table.insert(pool.species, species)
		species.topFitness = file:read("*number")
		species.staleness = file:read("*number")
		local numGenomes = file:read("*number")
		for g=1,numGenomes do
			local genome = newGenome()
			table.insert(species.genomes, genome)
			genome.fitness = file:read("*number")
			genome.maxneuron = file:read("*number")
			local line = file:read("*line")
			while line ~= "done" do
				genome.mutationRates[line] = file:read("*number")
				line = file:read("*line")
			
			local numGenes = file:read("*number")
			for n=1,numGenes do
				local gene = newGene()
				table.insert(genome.genes, gene)
				local enabled
				gene.into, gene.out, gene.weight, gene.innovation, enabled = file:read("*number", "*number", "*number", "*number", "*number")
				if enabled == 0:
					gene.enabled = false
				else
					gene.enabled = true
				
				
			
		
	
        file:close()
	
	while fitnessAlreadyMeasured() do
		nextGenome()
	
	initializeRun()
	pool.currentFrame = pool.currentFrame + 1

 
def loadPool()
	local filename = forms.gettext(saveLoadFile)
	loadFile(filename)


def playTop()
	local maxfitness = 0
	local maxs, maxg
	for s,species in pairs(pool.species) do
		for g,genome in pairs(species.genomes) do
			if genome.fitness > maxfitness:
				maxfitness = genome.fitness
				maxs = s
				maxg = g
			
		
	
	
	pool.currentSpecies = maxs
	pool.currentGenome = maxg
	pool.maxFitness = maxfitness
	forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
	initializeRun()
	pool.currentFrame = pool.currentFrame + 1
	return


def onExit()
	forms.destroy(form)


writeFile("temp.pool")

event.onexit(onExit)

form = forms.newform(200, 260, "Fitness")
maxFitnessLabel = forms.label(form, "Max Fitness: " .. math.floor(pool.maxFitness), 5, 8)
showNetwork = forms.checkbox(form, "Show Map", 5, 30)
showMutationRates = forms.checkbox(form, "Show M-Rates", 5, 52)
restartButton = forms.button(form, "Restart", initializePool, 5, 77)
saveButton = forms.button(form, "Save", savePool, 5, 102)
loadButton = forms.button(form, "Load", loadPool, 80, 102)
saveLoadFile = forms.textbox(form, Filename .. ".pool", 170, 25, nil, 5, 148)
saveLoadLabel = forms.label(form, "Save/Load:", 5, 129)
playTopButton = forms.button(form, "Play Top", playTop, 5, 170)
hideBanner = forms.checkbox(form, "Hide Banner", 5, 190)


while true do
	local backgroundColor = 0xD0FFFFFF
	if not forms.ischecked(hideBanner):
		gui.drawBox(0, 0, 300, 26, backgroundColor, backgroundColor)
	

	local species = pool.species[pool.currentSpecies]
	local genome = species.genomes[pool.currentGenome]
	
	if forms.ischecked(showNetwork):
		displayGenome(genome)
	
	
	if pool.currentFrame%5 == 0:
		evaluateCurrent()
	

	joypad.set(controller)

	getPositions()
	if marioX > rightmost:
		rightmost = marioX
		timeout = TimeoutConstant
	
	
	timeout = timeout - 1
	
	
	local timeoutBonus = pool.currentFrame / 4
	if timeout + timeoutBonus <= 0:
		local fitness = rightmost - pool.currentFrame / 2
		if gameinfo.getromname() == "Super Mario World (USA)" and rightmost > 4816:
			fitness = fitness + 1000
		
		if gameinfo.getromname() == "Super Mario Bros." and rightmost > 3186:
			fitness = fitness + 1000
		
		if fitness == 0:
			fitness = -1
		
		genome.fitness = fitness
		
		if fitness > pool.maxFitness:
			pool.maxFitness = fitness
			forms.settext(maxFitnessLabel, "Max Fitness: " .. math.floor(pool.maxFitness))
			writeFile("backup." .. pool.generation .. "." .. forms.gettext(saveLoadFile))
		
		
		console.writeline("Gen " .. pool.generation .. " species " .. pool.currentSpecies .. " genome " .. pool.currentGenome .. " fitness: " .. fitness)
		pool.currentSpecies = 1
		pool.currentGenome = 1
		while fitnessAlreadyMeasured() do
			nextGenome()
		
		initializeRun()
	

	local measured = 0
	local total = 0
	for _,species in pairs(pool.species) do
		for _,genome in pairs(species.genomes) do
			total = total + 1
			if genome.fitness ~= 0:
				measured = measured + 1
			
		
	
	if not forms.ischecked(hideBanner):
		gui.drawText(0, 0, "Gen " .. pool.generation .. " species " .. pool.currentSpecies .. " genome " .. pool.currentGenome .. " (" .. math.floor(measured/total*100) .. "%)", 0xFF000000, 11)
		gui.drawText(0, 12, "Fitness: " .. math.floor(rightmost - (pool.currentFrame) / 2 - (timeout + timeoutBonus)*2/3), 0xFF000000, 11)
		gui.drawText(100, 12, "Max Fitness: " .. math.floor(pool.maxFitness), 0xFF000000, 11)
	
		
	pool.currentFrame = pool.currentFrame + 1

	emu.frameadvance();
