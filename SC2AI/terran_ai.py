import sc2
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.ids.upgrade_id import UpgradeId

class TerranBot(BotAI):
    async def on_start(self):
        self.worker_count = 0
        self.supply_depot_count = 0
        self.refinery_count = 0
        self.expansion_locations_list = []
        self.barracks_count = 0
        self.last_supply_check = 0
        self.army_size = 0
        self.last_attack_time = 0
        self.attack_interval = 50  # Attack more frequently against hard AI
        self.scout_sent = False
        self.early_aggression = False
        self.scout_unit_tag = None
        self.last_scout_time = 0
        self.scout_interval = 200  # Increased scout interval to reduce worker loss
        self.defense_position = None
        self.last_defense_check = 0
        self.scout_target = None
        self.scout_start_time = 0
        self.scout_timeout = 100  # Timeout for scout to return if not reached target
        self.unit_weights = {
            UnitTypeId.MARINE: 1,
            UnitTypeId.MARAUDER: 2,
            UnitTypeId.MEDIVAC: 2,
            UnitTypeId.SIEGETANK: 3,
            UnitTypeId.VIKINGFIGHTER: 2,
            UnitTypeId.BANSHEE: 3,  # Added for harassment
            UnitTypeId.RAVEN: 3     # Added for support
        }

    async def on_step(self, iteration):
        # Get the command centers
        townhalls = self.townhalls
        if not townhalls:
            return
        cc = townhalls.first

        # Early game scouting
        if not self.scout_sent and self.workers.amount >= 12:
            self.scout_sent = True
            scout = self.workers.closest_to(self.enemy_start_locations[0])
            scout.attack(self.enemy_start_locations[0])
            self.scout_unit_tag = scout.tag

        # Process base and tech buildings only every few frames for efficiency
        if iteration % 5 == 0:
            await self.manage_economy(townhalls)
            await self.manage_supply(iteration)
            await self.manage_production(townhalls)
            await self.manage_upgrades()
            await self.manage_army(iteration)
            await self.manage_scouting(iteration)
            await self.manage_defense(iteration)

        # Process worker distribution every frame for better response
        await self.distribute_workers(townhalls)

    async def manage_defense(self, iteration):
        # Check for enemy units near our bases
        if iteration - self.last_defense_check > 10:
            self.last_defense_check = iteration
            
            for townhall in self.townhalls:
                # Check for enemy units within 20 units of our base
                enemy_units = self.enemy_units.closer_than(20, townhall)
                if enemy_units:
                    # Set defense position
                    self.defense_position = townhall.position
                    
                    # Pull workers to safety if under attack
                    if enemy_units.amount > 3:  # Only pull workers for significant threats
                        for worker in self.workers.closer_than(15, townhall):
                            worker.return_resource()
                            worker.move(townhall.position.towards(self.game_info.map_center, 5))

    async def manage_scouting(self, iteration):
        # Handle scout unit
        if self.scout_unit_tag:
            scout = self.units.find_by_tag(self.scout_unit_tag)
            if not scout or not scout.is_active:
                self.scout_unit_tag = None
                self.scout_sent = False
                self.scout_target = None
                self.scout_start_time = 0
            else:
                # Check if scout has timed out
                if iteration - self.scout_start_time > self.scout_timeout:
                    # Return scout to mining
                    mineral_field = self.mineral_field.closest_to(scout)
                    if mineral_field:
                        scout.return_resource()
                        scout.gather(mineral_field)
                        self.scout_unit_tag = None
                        self.scout_sent = False
                        self.scout_target = None
                        self.scout_start_time = 0
                # If scout has reached its target, return to mining
                elif self.scout_target and scout.position.distance_to(self.scout_target) < 5:
                    # Find nearest mineral field
                    mineral_field = self.mineral_field.closest_to(scout)
                    if mineral_field:
                        scout.return_resource()
                        scout.gather(mineral_field)
                        self.scout_unit_tag = None
                        self.scout_sent = False
                        self.scout_target = None
                        self.scout_start_time = 0

        # Periodic scouting
        if iteration - self.last_scout_time > self.scout_interval and not self.scout_unit_tag:
            self.last_scout_time = iteration
            
            # Get available workers for scouting
            available_workers = self.workers.filter(lambda w: w.is_gathering and not w.is_carrying_minerals)
            
            if available_workers:
                scout = available_workers.first
                if scout:
                    # Scout enemy base or natural expansion
                    if self.enemy_structures:
                        self.scout_target = self.enemy_structures.first.position
                    else:
                        self.scout_target = self.enemy_start_locations[0]
                    scout.attack(self.scout_target)
                    self.scout_unit_tag = scout.tag
                    self.scout_start_time = iteration

    async def manage_economy(self, townhalls):
        cc = townhalls.first
        
        # Build SCVs if we have less than 22 per base (16 minerals + 6 gas)
        if self.workers.amount < min(22 * townhalls.amount, 70) and self.can_afford(UnitTypeId.SCV):
            for cc in townhalls.idle:
                cc.train(UnitTypeId.SCV)

        # Build refineries - limit to 2 per base initially
        if self.workers.amount >= 14:
            for cc in townhalls:
                # Get vespene geysers near this base
                vespene_geysers = self.vespene_geyser.closer_than(15, cc)
                # Get existing refineries
                refineries = self.structures(UnitTypeId.REFINERY)
                refineries_near_cc = refineries.closer_than(15, cc)
                
                # If we have fewer than 2 refineries near this CC, build more
                if len(refineries_near_cc) < 2:
                    for vespene in vespene_geysers:
                        # Check if there's already a refinery on this geyser
                        if not refineries.closer_than(1, vespene) and self.can_afford(UnitTypeId.REFINERY):
                            await self.build(UnitTypeId.REFINERY, vespene)
                            break

        # Build expansion when we can afford it and have enough workers
        if self.workers.amount >= 18 and self.can_afford(UnitTypeId.COMMANDCENTER):
            # Store expansion locations if not done yet
            if not hasattr(self, 'expansion_locations_list') or not self.expansion_locations_list:
                self.expansion_locations_list = list(self.expansion_locations.keys())
                
            # Get existing CC positions
            existing_cc_positions = {cc.position for cc in townhalls}
            
            # Find next expansion location
            for location in self.expansion_locations_list:
                if location not in existing_cc_positions and await self.can_place(UnitTypeId.COMMANDCENTER, location):
                    # Send a worker to build the expansion
                    workers = self.workers.gathering
                    if workers:
                        worker = workers.closest_to(location)
                        worker.build(UnitTypeId.COMMANDCENTER, location)
                        break

    async def manage_supply(self, iteration):
        # Check supply status periodically
        if not hasattr(self, 'last_supply_check'):
            self.last_supply_check = 0
            
        if iteration - self.last_supply_check > 10:
            self.last_supply_check = iteration
            
            # Calculate needed supply
            current_supply = self.supply_used
            max_supply = self.supply_cap
            
            # Calculate how many supply depots we need to reach 200
            supply_depots = self.structures(UnitTypeId.SUPPLYDEPOT).ready
            current_depot_supply = supply_depots.amount * 8  # Each depot provides 8 supply
            
            # Calculate how many more depots we need to reach 200
            target_supply = 200
            needed_supply = max(0, target_supply - current_depot_supply)
            depots_needed = (needed_supply + 7) // 8  # Round up division
            
            # Also consider current supply usage
            if current_supply > max_supply - 8:  # If we're close to supply cap
                depots_needed = max(depots_needed, 1)  # Build at least one depot
            
            # Build depots if needed
            if depots_needed > 0 and self.can_afford(UnitTypeId.SUPPLYDEPOT):
                cc = self.townhalls.first
                
                # Find positions near command center in a line
                pos = cc.position.towards(self.game_info.map_center, 8)
                for i in range(depots_needed):
                    depot_pos = pos.offset(Point2((i * 3, 0)))
                    if await self.can_place(UnitTypeId.SUPPLYDEPOT, depot_pos):
                        # Send a worker to build the depot
                        workers = self.workers.gathering
                        if workers:
                            worker = workers.closest_to(depot_pos)
                            worker.build(UnitTypeId.SUPPLYDEPOT, depot_pos)
                            break

    async def manage_production(self, townhalls):
        cc = townhalls.first
        
        # Build barracks if we have supply depots
        supply_depots = self.structures(UnitTypeId.SUPPLYDEPOT).ready
        barracks = self.structures(UnitTypeId.BARRACKS)
        
        if supply_depots and barracks.amount < 4 and self.can_afford(UnitTypeId.BARRACKS):  # Build more barracks
            # Build barracks near supply depots
            depot = supply_depots.first
            pos = depot.position.towards(self.game_info.map_center, 4)
            await self.build(UnitTypeId.BARRACKS, near=pos)
        
        # Build factory when we have enough barracks
        factory = self.structures(UnitTypeId.FACTORY)
        if barracks.amount >= 2 and factory.amount < 2 and self.can_afford(UnitTypeId.FACTORY):
            pos = barracks.first.position.towards(self.game_info.map_center, 4)
            await self.build(UnitTypeId.FACTORY, near=pos)

        # Build starport when we have factory
        starport = self.structures(UnitTypeId.STARPORT)
        if factory.amount >= 1 and starport.amount < 2 and self.can_afford(UnitTypeId.STARPORT):  # Build more starports
            pos = factory.first.position.towards(self.game_info.map_center, 4)
            await self.build(UnitTypeId.STARPORT, near=pos)

        # Train units based on composition
        for barracks in barracks.idle:
            if self.can_afford(UnitTypeId.MARINE):
                barracks.train(UnitTypeId.MARINE)
            elif self.can_afford(UnitTypeId.MARAUDER):
                barracks.train(UnitTypeId.MARAUDER)

        for factory in factory.idle:
            if self.can_afford(UnitTypeId.SIEGETANK):
                factory.train(UnitTypeId.SIEGETANK)

        for starport in starport.idle:
            if self.can_afford(UnitTypeId.MEDIVAC):
                starport.train(UnitTypeId.MEDIVAC)
            elif self.can_afford(UnitTypeId.VIKINGFIGHTER):
                starport.train(UnitTypeId.VIKINGFIGHTER)
            elif self.can_afford(UnitTypeId.BANSHEE):
                starport.train(UnitTypeId.BANSHEE)
            elif self.can_afford(UnitTypeId.RAVEN):
                starport.train(UnitTypeId.RAVEN)

    async def manage_upgrades(self):
        # Research upgrades when we can afford them
        engineering_bays = self.structures(UnitTypeId.ENGINEERINGBAY)
        if engineering_bays.ready:
            if self.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1):
                engineering_bays.first.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)
            elif self.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL1):
                engineering_bays.first.research(UpgradeId.TERRANINFANTRYARMORSLEVEL1)

    async def manage_army(self, iteration):
        # Group army units
        army_units = self.units.filter(lambda unit: unit.type_id in [UnitTypeId.MARINE, UnitTypeId.MARAUDER, UnitTypeId.MEDIVAC, UnitTypeId.SIEGETANK, UnitTypeId.VIKINGFIGHTER])
        
        # Handle defense if needed
        if self.defense_position:
            for unit in army_units:
                unit.attack(self.defense_position)
            return
        
        if army_units.amount >= 15:  # Reduced minimum army size for more aggressive play
            # Attack periodically
            if iteration - self.last_attack_time > self.attack_interval:
                self.last_attack_time = iteration
                
                # Find enemy base or structures
                if self.enemy_structures:
                    target = self.enemy_structures.first.position
                else:
                    target = self.enemy_start_locations[0]
                
                # Group units and attack
                for unit in army_units:
                    unit.attack(target)

    async def distribute_workers(self, townhalls):
        cc = townhalls.first
        
        # Handle refineries - ensure exactly 3 workers per refinery
        refineries = self.structures(UnitTypeId.REFINERY).ready
        
        for refinery in refineries:
            # Maintain exactly 3 workers per refinery
            assigned = self.workers.filter(lambda w: w.order_target == refinery.tag)
            
            if len(assigned) > 3:
                # Move excess workers to minerals
                for worker in assigned[3:]:
                    worker.return_resource()
                    worker.gather(self.mineral_field.closest_to(worker))
                    
            elif len(assigned) < 3:
                # Get available mineral workers
                mineral_workers = self.workers.filter(lambda w: w.is_carrying_minerals or w.order_target in [m.tag for m in self.mineral_field])
                
                # Assign workers from minerals to gas
                available_workers = mineral_workers.sorted_by_distance_to(refinery)
                for worker in available_workers[:3 - len(assigned)]:
                    worker.return_resource()
                    worker.gather(refinery)
        
        # Handle minerals - try to maintain 2 workers per patch
        mineral_fields = self.mineral_field
        
        for townhall in townhalls:
            nearby_minerals = mineral_fields.closer_than(10, townhall)
            
            for mineral in nearby_minerals:
                # Count workers mining this patch
                mining_workers = self.workers.filter(lambda w: w.order_target == mineral.tag)
                
                if len(mining_workers) > 2:
                    # Reassign excess workers
                    for worker in mining_workers[2:]:
                        # Check if we have understaffed minerals
                        understaffed_minerals = [m for m in nearby_minerals if len(self.workers.filter(lambda w: w.order_target == m.tag)) < 2]
                        
                        if understaffed_minerals:
                            worker.return_resource()
                            worker.gather(understaffed_minerals[0])

def main():
    run_game(
        sc2.maps.get("Simple64"),
        [
            Bot(Race.Terran, TerranBot()),
            Computer(Race.Zerg, Difficulty.Hard)  # Changed to Hard difficulty
        ],
        realtime=False,
    )

if __name__ == "__main__":
    main() 