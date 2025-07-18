{{ ... }}

    def _calculate_order_slices(self) -> List[Dict]:
        """Calculate optimal order slices based on current market conditions."""
        if self.remaining_quantity <= 0:
            return []
        
        slices = []
        remaining = self.remaining_quantity
        
        # Calculate aggressive price based on urgency and market conditions
        if self.original_order.side == OrderSide.BUY:
            reference_price = self.market_state.best_ask if self.market_state.best_ask > 0 else self.original_order.price
            aggressive_price = reference_price * (1 + self.max_slippage)
            
            # Adjust based on urgency
            if self.urgency > 0.7:  # Very urgent - pay up to get filled
                aggressive_price = min(
                    aggressive_price * (1 + Decimal('0.0005')),  # 5bps more
                    reference_price * (1 + self.max_slippage * 2)  # But respect max slippage
                )
        else:  # SELL
            reference_price = self.market_state.best_bid if self.market_state.best_bid > 0 else self.original_order.price
            aggressive_price = reference_price * (1 - self.max_slippage)
            
            if self.urgency > 0.7:  # Very urgent - pay down to get filled
                aggressive_price = max(
                    aggressive_price * (1 - Decimal('0.0005')),  # 5bps less
                    reference_price * (1 - self.max_slippage * 2)  # But respect max slippage
                )
        
        # Calculate slice size based on market depth and volatility
        slice_size = self._calculate_optimal_slice_size()
        
        # First try: aggressive limit order
        if self.original_order.order_type == OrderType.LIMIT:
            slices.append({
                'quantity': min(slice_size, remaining),
                'price': aggressive_price,
                'order_type': OrderType.LIMIT,
                'time_in_force': TimeInForce.IOC,
                'aggressive': True
            })
            remaining -= slices[-1]['quantity']
        
        # Second try: dark pool (if enabled and size is appropriate)
        if (remaining > 0 and self.dark_pool_enabled and 
            self.remaining_quantity >= self.min_slice_size * 2):
            dp_quantity = min(remaining, self.max_slice_size * 2)  # Can be larger for dark pools
            slices.append({
                'quantity': dp_quantity,
                'price': reference_price,  # Mid-price or better
                'order_type': OrderType.LIMIT,
                'time_in_force': TimeInForce.DAY,
                'dark_pool': True,
                'aggressive': False
            })
            remaining -= dp_quantity
        
        # Third try: sweep the book if very urgent
        if remaining > 0 and self.urgency > 0.8 and \
           len(self.market_state.order_book.get('asks' if self.original_order.side == OrderSide.BUY else 'bids', [])) > 0:
            # Calculate quantity needed to sweep through price levels
            levels = self.market_state.order_book['asks' if self.original_order.side == OrderSide.BUY else 'bids']
            sweep_quantity = Decimal('0')
            sweep_price = Decimal('0')
            
            for price, qty in levels:
                if sweep_quantity >= remaining:
                    break
                sweep_quantity += Decimal(str(qty))
                sweep_price = Decimal(str(price))
            
            if sweep_quantity > 0:
                sweep_size = min(remaining, sweep_quantity)
                slices.append({
                    'quantity': sweep_size,
                    'price': sweep_price,
                    'order_type': OrderType.MARKET if self.urgency > 0.9 else OrderType.LIMIT,
                    'time_in_force': TimeInForce.IOC,
                    'aggressive': True,
                    'sweep': True
                })
                remaining -= sweep_size
        
        # Final fallback: regular limit order
        if remaining > 0:
            slices.append({
                'quantity': remaining,
                'price': aggressive_price,
                'order_type': OrderType.MARKET if self.urgency > 0.8 else OrderType.LIMIT,
                'time_in_force': TimeInForce.IOC,
                'aggressive': self.urgency > 0.5
            })
        
        return slices
    
    def _calculate_optimal_slice_size(self) -> Decimal:
        """Calculate the optimal slice size based on current market conditions."""
        # Base slice size based on urgency
        base_size = self.min_slice_size + (self.max_slice_size - self.min_slice_size) * Decimal(str(self.urgency))
        
        # Adjust for volatility
        if self.volatility_adaptive:
            # Higher volatility → smaller slices
            vol_factor = max(0.3, 1.0 - float(self.volatility) * 2)
            base_size *= Decimal(str(vol_factor))
        
        # Adjust for remaining quantity
        remaining_pct = float(self.remaining_quantity) / float(self.original_order.quantity)
        if remaining_pct < 0.2:  # Last 20% can be more aggressive
            base_size = min(base_size * Decimal('1.5'), self.max_slice_size)
        
        # Ensure we don't exceed max participation
        market_volume = self.market_state.volume_24h or (self.remaining_quantity * 1000)  # Fallback
        if market_volume > 0:
            max_by_participation = market_volume * self.max_participation
            base_size = min(base_size, max_by_participation)
        
        # Final adjustments
        base_size = max(self.min_slice_size, min(base_size, self.max_slice_size, self.remaining_quantity))
        
        # Round to lot size if needed
        lot_size = self.market_data.get('lot_size', Decimal('0.00000001'))
        if lot_size > 0:
            base_size = (base_size / lot_size).quantize(Decimal('1.'), rounding=ROUND_DOWN) * lot_size
        
        return base_size
    
    async def _submit_orders(self, order_slices: List[Dict]):
        """Submit orders based on the calculated slices."""
        if not order_slices:
            return
        
        for slice_def in order_slices:
            if self.remaining_quantity <= 0 or self.cancellation_event.is_set():
                break
                
            try:
                # Skip if slice size is too small
                if slice_def['quantity'] < self.min_slice_size * Decimal('0.1'):  # 10% of min size
                    continue
                
                # Create order
                order = Order(
                    order_id=f"sniper_{self.original_order.order_id}_{len(self.working_orders)}",
                    client_order_id=f"{self.original_order.client_order_id}_sniper_{len(self.working_orders)}",
                    symbol=self.original_order.symbol,
                    side=self.original_order.side,
                    order_type=slice_def['order_type'],
                    quantity=slice_def['quantity'],
                    price=slice_def['price'],
                    time_in_force=slice_def.get('time_in_force', TimeInForce.IOC),
                    status=OrderStatus.NEW,
                    timestamp=datetime.now(timezone.utc),
                    parent_order_id=self.original_order.order_id,
                    metadata={
                        'aggressive': slice_def.get('aggressive', False),
                        'dark_pool': slice_def.get('dark_pool', False),
                        'sweep': slice_def.get('sweep', False)
                    }
                )
                
                # Submit order
                if slice_def.get('dark_pool', False):
                    # Special handling for dark pool orders
                    order_id = await self._submit_dark_pool_order(order)
                    if order_id:
                        order.order_id = order_id
                        order.status = OrderStatus.NEW
                        self.dark_pool_orders[order_id] = order
                        self.working_orders.append(order)
                else:
                    # Regular exchange order
                    result = await self.exchange_adapter.submit_order(
                        symbol=order.symbol,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=order.quantity,
                        price=order.price,
                        time_in_force=order.time_in_force,
                        client_order_id=order.client_order_id
                    )
                    
                    # Update order with exchange response
                    if result:
                        order.order_id = result.get('order_id', order.order_id)
                        order.status = OrderStatus[result.get('status', 'NEW')]
                        
                        # If order was filled immediately, update execution metrics
                        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                            filled_qty = Decimal(str(result.get('filled_quantity', '0')))
                            fill_price = Decimal(str(result.get('filled_price', '0')))
                            if filled_qty > 0 and fill_price > 0:
                                self._on_order_fill(order, filled_qty, fill_price)
                        
                        # Add to working orders if not filled
                        if order.status == OrderStatus.NEW:
                            self.working_orders.append(order)
            
            except Exception as e:
                logger.error(f"Error submitting order slice: {str(e)}", exc_info=True)
                self.retry_count += 1
                if self.retry_count > self.max_retries:
                    raise
    
    async def _submit_dark_pool_order(self, order: Order) -> Optional[str]:
        """Submit an order to a dark pool."""
        try:
            # In a real implementation, this would connect to a dark pool ATS
            # For simulation, we'll use a random fill probability
            if random.random() < self.dark_pool_fill_probability:
                # Simulate a fill at mid-price
                fill_price = (self.market_state.best_bid + self.market_state.best_ask) / 2
                
                # Simulate partial fill (30-100%)
                fill_ratio = 0.3 + random.random() * 0.7  # 30-100%
                filled_qty = order.quantity * Decimal(str(fill_ratio))
                
                # Update execution metrics
                self._on_order_fill(order, filled_qty, fill_price)
                
                # Return None to indicate the order was filled and shouldn't be tracked
                return None
            else:
                # Return a simulated order ID for the working order
                return f"dp_{order.client_order_id}"
                
        except Exception as e:
            logger.error(f"Error submitting to dark pool: {str(e)}", exc_info=True)
            return None
    
    def _on_order_fill(self, order: Order, filled_qty: Decimal, fill_price: Decimal):
        """Handle order fill and update execution metrics."""
        try:
            # Update filled quantity and remaining quantity
            filled_qty = min(filled_qty, order.remaining_quantity)
            order.filled_quantity = (order.filled_quantity or Decimal('0')) + filled_qty
            order.remaining_quantity = order.quantity - order.filled_quantity
            
            # Update order status
            if order.remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Update execution metrics
            self.executed_volume += filled_qty
            self.executed_notional += filled_qty * fill_price
            self.executed_vwap = self.executed_notional / self.executed_volume if self.executed_volume > 0 else Decimal('0')
            self.remaining_quantity = max(Decimal('0'), self.original_order.quantity - self.executed_volume)
            
            # Record fill
            fill = {
                'order_id': order.order_id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'price': fill_price,
                'quantity': filled_qty,
                'timestamp': datetime.now(timezone.utc),
                'liquidity': 'MAKER' if not order.metadata.get('aggressive', False) else 'TAKER',
                'venue': 'DARK_POOL' if order.metadata.get('dark_pool', False) else 'LIT'
            }
            self.fill_history.append(fill)
            
            logger.info(f"Fill: {fill}")
            
        except Exception as e:
            logger.error(f"Error processing order fill: {str(e)}", exc_info=True)
            raise
    
    async def _optimize_working_orders(self):
        """Optimize working orders by cancelling and replacing suboptimal ones."""
        if not self.working_orders:
            return
        
        # Check each working order
        orders_to_cancel = []
        current_time = datetime.now(timezone.utc)
        
        for order in self.working_orders:
            try:
                # Skip dark pool orders (handled separately)
                if order.metadata.get('dark_pool', False):
                    continue
                
                # Check if order is too old (stale)
                order_age = (current_time - order.timestamp).total_seconds()
                max_age = self.refresh_interval * 3  # 3x refresh interval
                
                if order_age > max_age:
                    logger.debug(f"Cancelling stale order {order.order_id} (age: {order_age:.1f}s > {max_age:.1f}s)")
                    orders_to_cancel.append(order)
                    continue
                
                # Check if order price is no longer competitive
                if order.order_type == OrderType.LIMIT and not order.metadata.get('sweep', False):
                    if self.original_order.side == OrderSide.BUY:
                        best_ask = self.market_state.best_ask
                        if best_ask > 0 and order.price < best_ask * Decimal('0.999'):  # 1bp inside
                            orders_to_cancel.append(order)
                            continue
                    else:  # SELL
                        best_bid = self.market_state.best_bid
                        if best_bid > 0 and order.price > best_bid * Decimal('1.001'):  # 1bp inside
                            orders_to_cancel.append(order)
                            continue
                
                # For aggressive orders, check if they've been working too long without fills
                if order.metadata.get('aggressive', False) and order_age > self.refresh_interval * 1.5:
                    orders_to_cancel.append(order)
                    continue
                
            except Exception as e:
                logger.error(f"Error optimizing order {order.order_id}: {str(e)}", exc_info=True)
                # Mark for cancellation on error to be safe
                orders_to_cancel.append(order)
        
        # Cancel the orders
        if orders_to_cancel:
            await self._cancel_orders(orders_to_cancel)
    
    async def _cancel_orders(self, orders: List[Order]):
        """Cancel the specified orders."""
        if not orders:
            return
        
        cancellation_tasks = []
        for order in orders:
            if order.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
                if order.metadata.get('dark_pool', False):
                    # Special handling for dark pool orders
                    cancellation_tasks.append(self._cancel_dark_pool_order(order))
                else:
                    cancellation_tasks.append(
                        self.exchange_adapter.cancel_order(order.order_id)
                    )
        
        if cancellation_tasks:
            await asyncio.gather(*cancellation_tasks, return_exceptions=True)
        
        # Remove from working orders
        cancelled_ids = {o.order_id for o in orders}
        self.working_orders = [o for o in self.working_orders if o.order_id not in cancelled_ids]
    
    async def _cancel_dark_pool_order(self, order: Order):
        """Cancel an order in the dark pool."""
        # In a real implementation, this would send a cancellation to the dark pool
        # For simulation, we'll just remove it from our tracking
        if order.order_id in self.dark_pool_orders:
            del self.dark_pool_orders[order.order_id]
    
    async def _cancel_all_orders(self):
        """Cancel all working orders."""
        if not self.working_orders and not self.dark_pool_orders:
            return
        
        logger.info(f"Cancelling all working orders ({len(self.working_orders)} regular, "
                   f"{len(self.dark_pool_orders)} dark pool)")
        
        # Cancel regular orders
        if self.working_orders:
            await self._cancel_orders(self.working_orders.copy())
        
        # Cancel dark pool orders
        if self.dark_pool_orders:
            dark_pool_orders = list(self.dark_pool_orders.values())
            for order in dark_pool_orders:
                await self._cancel_dark_pool_order(order)
    
    def _monitor_execution(self):
        """Monitor execution and adjust strategy if needed."""
        # Check for adverse selection
        if self.anti_gaming:
            self._check_for_adverse_selection()
        
        # Adjust strategy based on volatility
        if self.volatility_adaptive:
            self._adjust_for_volatility()
        
        # Update dark pool fill probability
        self._update_dark_pool_probability()
    
    def _check_for_adverse_selection(self):
        """Check for signs of adverse selection or predatory trading."""
        if len(self.order_book_updates) < 5:  # Need some history
            return
        
        # Simple heuristic: count the number of times the market moved away after our orders
        adverse_moves = 0
        total_moves = 0
        
        for i in range(1, len(self.order_book_updates)):
            prev_time, prev_book = self.order_book_updates[i-1]
            curr_time, curr_book = self.order_book_updates[i]
            
            if not prev_book or not curr_book or not prev_book.get('bids') or not prev_book.get('asks'):
                continue
                
            prev_best_bid = Decimal(str(prev_book['bids'][0][0])) if prev_book['bids'] else Decimal('0')
            prev_best_ask = Decimal(str(prev_book['asks'][0][0])) if prev_book['asks'] else Decimal('inf')
            curr_best_bid = Decimal(str(curr_book['bids'][0][0])) if curr_book['bids'] else Decimal('0')
            curr_best_ask = Decimal(str(curr_book['asks'][0][0])) if curr_book['asks'] else Decimal('inf')
            
            # Check for adverse moves (market moving away after our orders)
            if self.original_order.side == OrderSide.BUY and curr_best_ask > prev_best_ask * Decimal('1.0001'):
                adverse_moves += 1
                total_moves += 1
            elif self.original_order.side == OrderSide.SELL and curr_best_bid < prev_best_bid * Decimal('0.9999'):
                adverse_moves += 1
                total_moves += 1
            elif self.original_order.side == OrderSide.BUY and curr_best_ask < prev_best_ask * Decimal('0.9999'):
                total_moves += 1
            elif self.original_order.side == OrderSide.SELL and curr_best_bid > prev_best_bid * Decimal('1.0001'):
                total_moves += 1
        
        # Update predation score (exponential moving average)
        if total_moves > 0:
            adverse_ratio = adverse_moves / total_moves
            alpha = 0.2  # Smoothing factor
            self.predation_score = alpha * adverse_ratio + (1 - alpha) * self.predation_score
            
            # If predation score is high, adjust strategy
            if self.predation_score > 0.6:
                logger.warning(f"High predation detected (score: {self.predation_score:.2f}). "
                             "Reducing urgency and slice sizes.")
                self.urgency = max(0.3, self.urgency * 0.8)  # Reduce urgency
                self.max_slice_size *= Decimal('0.8')  # Reduce slice size
    
    def _adjust_for_volatility(self):
        """Adjust strategy based on current volatility regime."""
        if self.volatility_regime == 'EXTREME':
            # In extreme volatility, be very conservative
            self.urgency = max(0.3, self.urgency * 0.7)
            self.max_slice_size = max(
                self.min_slice_size,
                self.max_slice_size * Decimal('0.5')
            )
        elif self.volatility_regime == 'HIGH':
            # In high volatility, be somewhat conservative
            self.urgency = max(0.5, self.urgency * 0.9)
            self.max_slice_size = max(
                self.min_slice_size,
                self.max_slice_size * Decimal('0.8')
            )
        elif self.volatility_regime == 'LOW':
            # In low volatility, can be more aggressive
            self.urgency = min(1.0, self.urgency * 1.1)
            self.max_slice_size = min(
                self.original_order.quantity,
                self.max_slice_size * Decimal('1.2')
            )
    
    def _create_execution_result(self) -> ExecutionResult:
        """Create an execution result with performance metrics."""
        self.end_time = datetime.now(timezone.utc)
        
        # Calculate VWAP and TWAP
        vwap = self.executed_notional / self.executed_volume if self.executed_volume > 0 else Decimal('0')
        
        # Calculate arrival price (price at start of execution)
        arrival_price = self.original_order.price
        
        # Calculate implementation shortfall
        benchmark_price = arrival_price
        if self.original_order.side == OrderSide.BUY:
            implementation_shortfall = (vwap - benchmark_price) / benchmark_price * Decimal('10000')  # in bps
        else:  # SELL
            implementation_shortfall = (benchmark_price - vwap) / benchmark_price * Decimal('10000')  # in bps
        
        # Calculate participation rate
        duration = (self.end_time - self.start_time).total_seconds()
        avg_volume_rate = float(self.market_state.volume_24h) / (24 * 60 * 60)  # Volume per second
        participation_rate = float(self.executed_volume) / (avg_volume_rate * duration) if duration > 0 and avg_volume_rate > 0 else 0.0
        
        # Create execution result
        result = ExecutionResult(
            order_id=self.original_order.order_id,
            symbol=self.original_order.symbol,
            side=self.original_order.side,
            order_type=self.original_order.order_type,
            quantity_ordered=self.original_order.quantity,
            quantity_executed=self.executed_volume,
            avg_execution_price=vwap,
            arrival_price=arrival_price,
            benchmark_price=benchmark_price,
            implementation_shortfall_bps=implementation_shortfall,
            participation_rate=Decimal(str(participation_rate)),
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=Decimal(str(duration)),
            fills=self.fill_history,
            metadata={
                'volatility_regime': self.volatility_regime,
                'urgency': self.urgency,
                'predation_score': self.predation_score,
                'dark_pool_fill_probability': self.dark_pool_fill_probability,
                'volatility': self.volatility
            }
        )
        
        logger.info(f"Execution completed: {result}")
        return result
