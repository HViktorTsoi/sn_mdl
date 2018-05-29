<template>
  <div>
    真实数据集拟合
  </div>
</template>
<script>
import IEcharts from 'vue-echarts-v3/src/full.js'
export default {
  name: 'comparison',
  components: {
    IEcharts
  },
  data () {
    return {
    }
  },
  computed: {
  },
  mounted () {
  },
  methods: {
  }
}
</script>
<style lang="scss" scoped>
.chart-container {
  height: 320px;
}
.el-form-item {
  margin-bottom: 3px;
  .query-time {
    width: 80px;
  }
}
.detail-info {
  .el-row {
    padding-top: 10px;
    padding-bottom: 10px;
    padding-left: 10px;
    border-bottom: 1px solid #eee;
  }
}
</style>
