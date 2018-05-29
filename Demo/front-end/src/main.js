import Vue from 'vue'
import VueRouter from 'vue-router'
import Vuex from 'vuex'
import axios from 'axios'

import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

import App from './App'

import store from './store'

Vue.config.productionTip = false

Vue.use(Vuex)
Vue.use(VueRouter)
Vue.use(ElementUI)
Vue.prototype.$http = axios

// 配置路由
var routes = [
  {
    path: '/',
    redirect: 'evolution'
  },
  {
    path: '/evolution',
    component: require('./components/Evolution').default,
    name: 'evolution'
  },
  {
    path: '/comparison',
    component: require('./components/Comparison').default,
    name: 'comparison'
  }
]
var router = new VueRouter({
  routes: routes
})

/* eslint-disable no-new */
new Vue({
  el: '#app',
  template: '<App/>',
  components: { App },
  store,
  router
})
