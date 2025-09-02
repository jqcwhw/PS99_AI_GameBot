#!/usr/bin/env python3
"""
PS99 Zones and Worlds API Integration
Real-time zone/world data for AI learning and navigation
"""

import requests
import json
import logging
from pathlib import Path

def add_ps99_zones_worlds_endpoints(app):
    """Add PS99 zones and worlds API endpoints to Flask app"""
    
    @app.route('/api/ps99/zones')
    def get_ps99_zones():
        """Get PS99 zones data from API with AI integration"""
        try:
            game_bot = app.config['GAME_BOT']
            
            # Initialize PS99 API integration if not present
            if not hasattr(game_bot, 'ps99_api'):
                from core.ps99_api_integration import initialize_ps99_integration
                initialize_ps99_integration(game_bot)
            
            # Get zones data through integrated API
            zones_data = game_bot.ps99_api.get_ps99_data('zones')
            
            return app.response_class(
                response=json.dumps({
                    'success': True, 
                    'zones': zones_data,
                    'count': len(zones_data),
                    'source': 'ps99.biggamesapi.io'
                }),
                status=200,
                mimetype='application/json'
            )
        except Exception as e:
            return app.response_class(
                response=json.dumps({'success': False, 'message': str(e)}),
                status=500,
                mimetype='application/json'
            )
    
    @app.route('/api/ps99/worlds')
    def get_ps99_worlds():
        """Get PS99 worlds data from API with AI integration"""
        try:
            game_bot = app.config['GAME_BOT']
            
            # Initialize PS99 API integration if not present
            if not hasattr(game_bot, 'ps99_api'):
                from core.ps99_api_integration import initialize_ps99_integration
                initialize_ps99_integration(game_bot)
            
            # Get worlds data through integrated API
            worlds_data = game_bot.ps99_api.get_ps99_data('worlds')
            
            return app.response_class(
                response=json.dumps({
                    'success': True, 
                    'worlds': worlds_data,
                    'count': len(worlds_data),
                    'source': 'ps99.biggamesapi.io'
                }),
                status=200,
                mimetype='application/json'
            )
        except Exception as e:
            return app.response_class(
                response=json.dumps({'success': False, 'message': str(e)}),
                status=500,
                mimetype='application/json'
            )
    
    @app.route('/api/ps99/validate-location', methods=['POST'])
    def validate_ps99_location():
        """Validate current game location against PS99 API data"""
        try:
            data = app.get_json()
            location_name = data.get('location', '')
            location_type = data.get('type', 'zone')  # zone or world
            
            game_bot = app.config['GAME_BOT']
            
            # Initialize PS99 API integration if not present
            if not hasattr(game_bot, 'ps99_api'):
                from core.ps99_api_integration import initialize_ps99_integration
                initialize_ps99_integration(game_bot)
            
            # Validate location
            if location_type == 'zone':
                location_info = game_bot.ps99_api.find_zone_by_name(location_name)
            else:
                location_info = game_bot.ps99_api.find_world_by_name(location_name)
            
            if location_info:
                # Create AI knowledge update
                knowledge_update = game_bot.ps99_api.create_ai_knowledge_update(
                    'location_entered', 
                    {
                        'location_name': location_name,
                        'location_type': location_type,
                        'location_info': location_info
                    }
                )
                
                # Add to AI learning system if available
                if hasattr(game_bot, 'learning_system'):
                    game_bot.learning_system.add_experience(knowledge_update)
                
                return app.response_class(
                    response=json.dumps({
                        'success': True,
                        'valid': True,
                        'location_info': location_info,
                        'ai_update': 'Knowledge updated with validated location'
                    }),
                    status=200,
                    mimetype='application/json'
                )
            else:
                return app.response_class(
                    response=json.dumps({
                        'success': True,
                        'valid': False,
                        'message': f'Location {location_name} not found in PS99 API'
                    }),
                    status=200,
                    mimetype='application/json'
                )
                
        except Exception as e:
            return app.response_class(
                response=json.dumps({'success': False, 'message': str(e)}),
                status=500,
                mimetype='application/json'
            )
    
    @app.route('/api/ps99/validate-hatch', methods=['POST'])
    def validate_ps99_hatch():
        """Validate hatched pet against PS99 API data"""
        try:
            data = app.get_json()
            pet_name = data.get('pet_name', '')
            egg_name = data.get('egg_name', '')
            
            game_bot = app.config['GAME_BOT']
            
            # Initialize PS99 API integration if not present
            if not hasattr(game_bot, 'ps99_api'):
                from core.ps99_api_integration import initialize_ps99_integration
                initialize_ps99_integration(game_bot)
            
            # Validate the hatch
            valid_hatch, pet_info = game_bot.ps99_api.validate_hatched_pet(pet_name, egg_name)
            
            # Create AI knowledge update
            knowledge_update = game_bot.ps99_api.create_ai_knowledge_update(
                'pet_hatched',
                {
                    'pet_name': pet_name,
                    'egg_name': egg_name,
                    'valid_hatch': valid_hatch,
                    'pet_info': pet_info
                }
            )
            
            # Add to AI learning system if available
            if hasattr(game_bot, 'learning_system'):
                game_bot.learning_system.add_experience(knowledge_update)
            
            return app.response_class(
                response=json.dumps({
                    'success': True,
                    'valid_hatch': valid_hatch,
                    'pet_info': pet_info,
                    'knowledge_update': 'AI learning updated with hatch validation'
                }),
                status=200,
                mimetype='application/json'
            )
            
        except Exception as e:
            return app.response_class(
                response=json.dumps({'success': False, 'message': str(e)}),
                status=500,
                mimetype='application/json'
            )
    
    @app.route('/api/ps99/api-stats')
    def get_ps99_api_stats():
        """Get PS99 API integration statistics"""
        try:
            game_bot = app.config['GAME_BOT']
            
            if hasattr(game_bot, 'ps99_api'):
                stats = game_bot.ps99_api.get_api_statistics()
                return app.response_class(
                    response=json.dumps({
                        'success': True,
                        'stats': stats
                    }),
                    status=200,
                    mimetype='application/json'
                )
            else:
                return app.response_class(
                    response=json.dumps({
                        'success': False,
                        'message': 'PS99 API integration not initialized'
                    }),
                    status=200,
                    mimetype='application/json'
                )
                
        except Exception as e:
            return app.response_class(
                response=json.dumps({'success': False, 'message': str(e)}),
                status=500,
                mimetype='application/json'
            )